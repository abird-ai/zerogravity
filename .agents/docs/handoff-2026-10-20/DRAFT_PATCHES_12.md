# Draft Patches — volume 12

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-hf2b-attention-paired-20261020/frozen-prepared/NEW_PREPARATION.patch

Original UTF-8 bytes: 174133; SHA256 `e8d6897bf2fb73ccb4741a8921f53568393fa6b3ddfcc6e8bbe55c79c6c360d6`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ CAPTURE_INTEGRATION.review.diff
@@ -0,0 +1,288 @@
+--- accepted-capture
++++ paired-capture
+@@ -14,6 +14,13 @@
+ import resource
+ import sys
+ import time
++
++# Pin the four allowed CPUs before importing the original Torch runtime.
++allowed = [int(value) for value in os.environ["ZG_PROBE_ALLOWED_CPUS"].split(",")]
++assert len(allowed) == len(set(allowed)) == 4 and set(allowed) <= os.sched_getaffinity(0)
++os.sched_setaffinity(0, allowed)
++MODE = os.environ["ZG_ATTENTION_MODE"]
++assert MODE in ("plain", "observed")
+ 
+ import torch
+ from safetensors import safe_open
+@@ -250,115 +257,147 @@
+     handles.append(module.register_forward_pre_hook(lambda _module, args: remember(name, args[0])))
+ 
+ 
+-language = model.model.language_model
+-layer0 = language.layers[0]
+-output_hook(language.embed_tokens, "embedding")
+-output_hook(layer0.input_layernorm, "layer-00.attention_input_norm")
+-output_hook(layer0.linear_attn.in_proj_qkv, "layer-00.gdn_qkv")
+-output_hook(layer0.linear_attn, "layer-00.attention_output")
+-input_hook(layer0.post_attention_layernorm, "layer-00.post_attention_residual")
+-output_hook(layer0.post_attention_layernorm, "layer-00.ffn_input_norm")
+-output_hook(layer0.mlp.gate_proj, "layer-00.ffn_gate")
+-output_hook(layer0.mlp.up_proj, "layer-00.ffn_up")
+-input_hook(layer0.mlp.down_proj, "layer-00.ffn_fused")
+-output_hook(layer0.mlp.down_proj, "layer-00.ffn_output")
+-output_hook(layer0, "layer-00.post_ffn_residual")
+-output_hook(language.norm, "final_norm")
+-
+-manifest = {
+-    "schema": "zerogravity.official-cpu-qwen35-reference.v2-hf2b",
+-    "source_pins_sha256": sha(pathlib.Path("/probe/source-pins.json")),
+-    "source_repo": source_receipt["repo"], "source_revision": source_receipt["revision"],
+-    "image": PINS["derived_image_id"], "base_image": PINS["base_image"],
+-    "runtime_source": runtime_source, "package_records": PINS["package_records"],
+-    "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "safetensors": importlib.metadata.version("safetensors"), "vllm": importlib.metadata.version("vllm"), "torch_num_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "cuda_available": torch.cuda.is_available(), "mkldnn_enabled": torch.backends.mkldnn.enabled, "default_dtype": str(torch.get_default_dtype()), "cpu_capability": torch.backends.cpu.get_cpu_capability(), "machine": platform.machine(), "processor": platform.processor()},
+-    "loader": {"model_class": type(model).__name__, "dtype": "torch.bfloat16", "attn_implementation": "eager", "use_kernels": False, "local_files_only": True, "loading_info": loading_info},
+-    "config": model.config.to_dict(), "tokenizer": {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer), "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id, "tokens": TOKENS, "token_strings": tokenizer.convert_ids_to_tokens(TOKENS), "decoded": tokenizer.decode(TOKENS, skip_special_tokens=False)},
+-    "head": {"tied_parameter_object": True, "source_has_lm_head_tensor": False, "embedding_name": "model.language_model.embed_tokens.weight", "output_dtype_rule": "official lm_head(hidden); no loss, no automatic FP32 upcast"},
+-    "protocol": "one cached official forward per fixed input token, no sampling or generation; first token uses chunk reference, next token recurrent reference",
+-    "limits": {"container_memory_max": pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip(), "container_cpu_max": pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip(), "network": "none", "source_mount": "read-only", "gpu_exposed": False},
+-    "scope": "Official Transformers eager CPU attention and slow PyTorch GDN operators only. No GPU FLA, vLLM inference, throughput, quantization, Rust bit parity, or trained27B qualification.",
+-    "profiles": {},
+-    "source_geometry": {"hidden": 2048, "ffn": 6144, "layers": 24, "gdn_layers": 18, "full_attention_layers": 6, "gdn_heads": 16, "gdn_head_width": 128, "kv_heads": 2, "kv_head_width": 256},
+-}
+-model_source = pathlib.Path(inspect.getsourcefile(official))
+-manifest["official_modeling_source"] = {"path": str(model_source), "sha256": sha(model_source)}
+-manifest["capture_script_sha256"] = sha(pathlib.Path(__file__))
+-
+-for profile in PROFILES:
+-    if profile == "source-36-f32-restored":
+-        with safe_open(source_file, framework="pt", device="cpu") as source_handle:
+-            for name in f32_names:
+-                parameter = model.get_parameter(name)
+-                parameter.data = source_handle.get_tensor(name).clone()
+-                assert parameter.dtype == torch.float32
+-        print("RESTORED_ORIGINAL_F32_PARAMETERS count=36", flush=True)
+-    profile_dir = OUTPUT / profile
+-    profile_dir.mkdir()
+-    audit = parameter_audit(model, source_file)
+-    write_json(profile_dir / "parameter-audit.json", audit)
+-    assert len(audit) == 632 and sum(row["loaded"] is None for row in audit) == 15
+-    dtype_counts = collections.Counter(row["loaded"]["dtype"] for row in audit if row["loaded"] is not None)
+-    print("PROFILE_AUDIT", profile, dict(dtype_counts), flush=True)
+-    profile_result = {"profile": profile, "parameter_audit_sha256": sha(profile_dir / "parameter-audit.json"), "source_parameter_count": len(audit), "actual_checkpoint_parameter_dtype_counts": dict(dtype_counts), "source_f32_parameter_names": f32_names, "restoration": "none; unmodified official requested-BF16 loader" if profile == PROFILES[0] else "only the 36 originally F32 checkpoint parameters restored to their exact source dtype and bytes; official operators unchanged", "steps": []}
+-    cache = None
+-    with torch.inference_mode():
+-        for step, token in enumerate(TOKENS):
+-            intermediates = {}
+-            calls = collections.Counter()
+-            signatures = {}
+-
+-            def observe_calls(frame, event, arg):
+-                if frame.f_code.co_filename != str(model_source):
+-                    return
+-                name = frame.f_code.co_name
+-                if event == "return" and name in {"causal_conv1d_fn", "causal_conv1d_update"}:
+-                    caller = frame.f_back
+-                    while caller is not None:
+-                        owner = caller.f_locals.get("self")
+-                        if isinstance(owner, official.Qwen3_5GatedDeltaNet):
+-                            if owner.layer_idx == 0:
+-                                remember("layer-00.gdn_conv_silu", arg.transpose(1, 2))
+-                            break
+-                        caller = caller.f_back
+-                    return
+-                if event != "call":
+-                    return
+-                if name in {"torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update", "eager_attention_forward"}:
+-                    key = name + ":" + str(frame.f_code.co_firstlineno)
+-                    calls[key] += 1
+-                    signatures[key] = {key: tensor_info(value) for key, value in frame.f_locals.items() if isinstance(value, torch.Tensor)}
+-
+-            step_start = time.time()
+-            sys.setprofile(observe_calls)
+-            try:
+-                result = model(input_ids=torch.tensor([[token]], dtype=torch.int64), past_key_values=cache, use_cache=True, return_dict=True)
+-            finally:
+-                sys.setprofile(None)
+-            cache = result.past_key_values
+-            expected = {("torch_chunk_gated_delta_rule" if step == 0 else "torch_recurrent_gated_delta_rule"): 18, ("causal_conv1d_fn" if step == 0 else "causal_conv1d_update"): 18, "eager_attention_forward": 6}
+-            observed_by_name = {key.split(":")[0]: value for key, value in calls.items()}
+-            assert observed_by_name == expected, (expected, observed_by_name)
+-            step_result = snapshot(cache, result.logits, profile_dir / f"step-{step}", profile, step, token)
+-            remember("final_logits", result.logits)
+-            remember("layer-00.gdn_recurrent_fp32", cache.layers[0].recurrent_states[0].permute(0, 1, 3, 2))
+-            assert len(intermediates) == 15
+-            intermediate_path = profile_dir / f"step-{step}" / "observed-intermediates.safetensors"
+-            save_file(intermediates, str(intermediate_path), metadata={"profile": profile, "step": str(step), "scope": "actual layer0/module input/output copies; no reconstructed arithmetic"})
+-            step_result["observed_intermediates"] = {"file": str(intermediate_path.relative_to(OUTPUT)), "sha256": sha(intermediate_path), "bytes": intermediate_path.stat().st_size, "tensors": {name: tensor_info(value) for name, value in intermediates.items()}, "hook_scope": "official module pre/post hooks, CPU tensor clone only; slow causal-conv source return observed with sys.setprofile; no replacement return"}
+-            step_result["observed_official_operator_calls"] = dict(calls)
+-            step_result["last_observed_operator_tensor_arguments"] = signatures
+-            step_result["diagnostic_elapsed_seconds_not_benchmark"] = time.time() - step_start
+-            write_json(profile_dir / f"step-{step}" / "manifest.json", step_result)
+-            profile_result["steps"].append(step_result)
+-            artifact_bound()
+-            print("REFERENCE_STEP_COMPLETE", profile, "step", step, "argmax", step_result["argmax_token_id"], "domains", len(step_result["semantic_domains"]), "seconds", step_result["diagnostic_elapsed_seconds_not_benchmark"], flush=True)
+-    manifest["profiles"][profile] = profile_result
+-    write_json(OUTPUT / "manifest.json", manifest)
+-    del result, cache
+-
+-for handle in handles:
+-    handle.remove()
++from attention_observation_hook import AttentionProjectionObserver
++attention = None
++try:
++    language = model.model.language_model
++    layer0 = language.layers[0]
++    output_hook(language.embed_tokens, "embedding")
++    output_hook(layer0.input_layernorm, "layer-00.attention_input_norm")
++    output_hook(layer0.linear_attn.in_proj_qkv, "layer-00.gdn_qkv")
++    output_hook(layer0.linear_attn, "layer-00.attention_output")
++    input_hook(layer0.post_attention_layernorm, "layer-00.post_attention_residual")
++    output_hook(layer0.post_attention_layernorm, "layer-00.ffn_input_norm")
++    output_hook(layer0.mlp.gate_proj, "layer-00.ffn_gate")
++    output_hook(layer0.mlp.up_proj, "layer-00.ffn_up")
++    input_hook(layer0.mlp.down_proj, "layer-00.ffn_fused")
++    output_hook(layer0.mlp.down_proj, "layer-00.ffn_output")
++    output_hook(layer0, "layer-00.post_ffn_residual")
++    output_hook(language.norm, "final_norm")
++
++    manifest = {
++        "schema": "zerogravity.official-cpu-qwen35-reference.v2-hf2b",
++        "source_pins_sha256": sha(pathlib.Path("/probe/source-pins.json")),
++        "source_repo": source_receipt["repo"], "source_revision": source_receipt["revision"],
++        "image": PINS["derived_image_id"], "base_image": PINS["base_image"],
++        "runtime_source": runtime_source, "package_records": PINS["package_records"],
++        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "safetensors": importlib.metadata.version("safetensors"), "vllm": importlib.metadata.version("vllm"), "torch_num_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "cuda_available": torch.cuda.is_available(), "mkldnn_enabled": torch.backends.mkldnn.enabled, "default_dtype": str(torch.get_default_dtype()), "cpu_capability": torch.backends.cpu.get_cpu_capability(), "machine": platform.machine(), "processor": platform.processor()},
++        "loader": {"model_class": type(model).__name__, "dtype": "torch.bfloat16", "attn_implementation": "eager", "use_kernels": False, "local_files_only": True, "loading_info": loading_info},
++        "config": model.config.to_dict(), "tokenizer": {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer), "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id, "tokens": TOKENS, "token_strings": tokenizer.convert_ids_to_tokens(TOKENS), "decoded": tokenizer.decode(TOKENS, skip_special_tokens=False)},
++        "head": {"tied_parameter_object": True, "source_has_lm_head_tensor": False, "embedding_name": "model.language_model.embed_tokens.weight", "output_dtype_rule": "official lm_head(hidden); no loss, no automatic FP32 upcast"},
++        "protocol": "one cached official forward per fixed input token, no sampling or generation; first token uses chunk reference, next token recurrent reference",
++        "limits": {"container_memory_max": pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip(), "container_cpu_max": pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip(), "network": "none", "source_mount": "read-only", "gpu_exposed": False},
++        "scope": "Official Transformers eager CPU attention and slow PyTorch GDN operators only. No GPU FLA, vLLM inference, throughput, quantization, Rust bit parity, or trained27B qualification.",
++        "profiles": {},
++        "source_geometry": {"hidden": 2048, "ffn": 6144, "layers": 24, "gdn_layers": 18, "full_attention_layers": 6, "gdn_heads": 16, "gdn_head_width": 128, "kv_heads": 2, "kv_head_width": 256},
++    }
++    model_source = pathlib.Path(inspect.getsourcefile(official))
++    manifest["official_modeling_source"] = {"path": str(model_source), "sha256": sha(model_source)}
++    manifest["capture_script_sha256"] = sha(pathlib.Path(__file__))
++
++    if MODE == "observed":
++        attention = AttentionProjectionObserver(model, official)
++    for profile in PROFILES:
++        if profile == "source-36-f32-restored":
++            with safe_open(source_file, framework="pt", device="cpu") as source_handle:
++                for name in f32_names:
++                    parameter = model.get_parameter(name)
++                    parameter.data = source_handle.get_tensor(name).clone()
++                    assert parameter.dtype == torch.float32
++            print("RESTORED_ORIGINAL_F32_PARAMETERS count=36", flush=True)
++        profile_dir = OUTPUT / profile
++        profile_dir.mkdir()
++        audit = parameter_audit(model, source_file)
++        write_json(profile_dir / "parameter-audit.json", audit)
++        assert len(audit) == 632 and sum(row["loaded"] is None for row in audit) == 15
++        dtype_counts = collections.Counter(row["loaded"]["dtype"] for row in audit if row["loaded"] is not None)
++        print("PROFILE_AUDIT", profile, dict(dtype_counts), flush=True)
++        profile_result = {"profile": profile, "parameter_audit_sha256": sha(profile_dir / "parameter-audit.json"), "source_parameter_count": len(audit), "actual_checkpoint_parameter_dtype_counts": dict(dtype_counts), "source_f32_parameter_names": f32_names, "restoration": "none; unmodified official requested-BF16 loader" if profile == PROFILES[0] else "only the 36 originally F32 checkpoint parameters restored to their exact source dtype and bytes; official operators unchanged", "steps": []}
++        cache = None
++        with torch.inference_mode():
++            for step, token in enumerate(TOKENS):
++                intermediates = {}
++                calls = collections.Counter()
++                signatures = {}
++
++                def observe_calls(frame, event, arg):
++                    if attention is not None:
++                        attention.observe_source_return(frame, event, arg)
++                    if frame.f_code.co_filename != str(model_source):
++                        return
++                    name = frame.f_code.co_name
++                    if event == "return" and name in {"causal_conv1d_fn", "causal_conv1d_update"}:
++                        caller = frame.f_back
++                        while caller is not None:
++                            owner = caller.f_locals.get("self")
++                            if isinstance(owner, official.Qwen3_5GatedDeltaNet):
++                                if owner.layer_idx == 0:
++                                    remember("layer-00.gdn_conv_silu", arg.transpose(1, 2))
++                                break
++                            caller = caller.f_back
++                        return
++                    if event != "call":
++                        return
++                    if name in {"torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update", "eager_attention_forward"}:
++                        key = name + ":" + str(frame.f_code.co_firstlineno)
++                        calls[key] += 1
++                        signatures[key] = {key: tensor_info(value) for key, value in frame.f_locals.items() if isinstance(value, torch.Tensor)}
++
++                if attention is not None:
++                    attention.begin_step(profile, step, token)
++                step_start = time.time()
++                sys.setprofile(observe_calls)
++                try:
++                    result = model(input_ids=torch.tensor([[token]], dtype=torch.int64), past_key_values=cache, use_cache=True, return_dict=True)
++                finally:
++                    sys.setprofile(None)
++                cache = result.past_key_values
++                expected = {("torch_chunk_gated_delta_rule" if step == 0 else "torch_recurrent_gated_delta_rule"): 18, ("causal_conv1d_fn" if step == 0 else "causal_conv1d_update"): 18, "eager_attention_forward": 6}
++                observed_by_name = {key.split(":")[0]: value for key, value in calls.items()}
++                assert observed_by_name == expected, (expected, observed_by_name)
++                step_result = snapshot(cache, result.logits, profile_dir / f"step-{step}", profile, step, token)
++                if attention is not None:
++                    attention.finish_step(profile_dir / f"step-{step}" / "attention-observation")
++                remember("final_logits", result.logits)
++                remember("layer-00.gdn_recurrent_fp32", cache.layers[0].recurrent_states[0].permute(0, 1, 3, 2))
++                assert len(intermediates) == 15
++                intermediate_path = profile_dir / f"step-{step}" / "observed-intermediates.safetensors"
++                save_file(intermediates, str(intermediate_path), metadata={"profile": profile, "step": str(step), "scope": "actual layer0/module input/output copies; no reconstructed arithmetic"})
++                step_result["observed_intermediates"] = {"file": str(intermediate_path.relative_to(OUTPUT)), "sha256": sha(intermediate_path), "bytes": intermediate_path.stat().st_size, "tensors": {name: tensor_info(value) for name, value in intermediates.items()}, "hook_scope": "official module pre/post hooks, CPU tensor clone only; slow causal-conv source return observed with sys.setprofile; no replacement return"}
++                step_result["observed_official_operator_calls"] = dict(calls)
++                step_result["last_observed_operator_tensor_arguments"] = signatures
++                step_result["diagnostic_elapsed_seconds_not_benchmark"] = time.time() - step_start
++                write_json(profile_dir / f"step-{step}" / "manifest.json", step_result)
++                profile_result["steps"].append(step_result)
++                artifact_bound()
++                print("REFERENCE_STEP_COMPLETE", profile, "step", step, "argmax", step_result["argmax_token_id"], "domains", len(step_result["semantic_domains"]), "seconds", step_result["diagnostic_elapsed_seconds_not_benchmark"], flush=True)
++        manifest["profiles"][profile] = profile_result
++        write_json(OUTPUT / "manifest.json", manifest)
++        del result, cache
++
++finally:
++    # Every installed hook is removed even after forward/capture/receipt failure.
++    primary = sys.exception()
++    removal = []
++    try:
++        sys.setprofile(None)
++    except BaseException as error:
++        removal.append(error)
++    if attention is not None:
++        try:
++            attention.close()
++        except BaseException as error:
++            removal.append(error)
++    attention = None  # Release observed module owners before archive validation.
++    for handle in reversed(handles):
++        try:
++            handle.remove()
++        except BaseException as error:
++            removal.append(error)
++    handles.clear()
++    if removal:
++        raise BaseExceptionGroup("capture and all hook removal failures",
++                                 ([primary] if primary is not None else []) + removal) from None
+ del model, embedding, language, layer0, intermediates
+ gc.collect()
+ assert verify_source() == source_receipt
+@@ -368,6 +407,14 @@
+ manifest["source_reverified_after_capture"] = True
+ manifest["elapsed_seconds_not_benchmark"] = time.time() - started
+ manifest["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
++write_json(pathlib.Path("/output/attention-mode.json"), {
++    "mode": MODE, "owned_hooks_removed": True, "profiler_retired": sys.getprofile() is None,
++    "cpu_affinity": sorted(os.sched_getaffinity(0)), "teacher_tokens": TOKENS,
++    "profiles": PROFILES, "observer_hook_sha256": sha(pathlib.Path("/probe/attention_observation_hook.py")),
++    "accepted_capture_sha256": "6ee140185766dee237ea4df802504ef7b7be80110f5566876a644414cf66049d",
++    "expected_attention_tensor_count": 216 if MODE == "observed" else 0,
++    "full_model_or_Rust_or_GPU_parity_qualified": False,
++})
+ write_json(OUTPUT / "manifest.json", manifest)
+ artifact_bound()
+ print("REFERENCE_CAPTURE_COMPLETE", json.dumps({"profiles": list(manifest["profiles"]), "steps_each": 2, "full_vocab": 248320, "semantic_domains_each": 42, "max_rss_kib": manifest["max_rss_kib"]}), flush=True)
--- /dev/null
+++ README.md
@@ -0,0 +1,90 @@
+# Original HF2B paired attention observation — ROOT executes
+
+- Owner: baseline_review. Parent: existing model-execution plan; ignored preparation only.
+- Source-only, unexecuted. No model/container/operator/test/build execution by author.
+- Exact original Qwen/Qwen3.5-2B revision15852e8c16360a2fea060d615a32b45270f8a8fc,
+  snapshot88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52.
+- CPU Transformers eager/slow GDN in the already acquired pinned image. This is
+  not vLLM inference or GPU evidence despite that image's installed packages.
+- No production source, shared plans, existing run bytes, weights or math changed.
+
+## Existing accepted reference
+
+`../model-execution-hf2b-cpu-reference-20261020/root-probe-2/captured-reference`
+is immutable. `parent-pins.json` pins its actual manifest/validation and original
+invocation, accepted capture/validator/source metadata, Batch's frozen attention
+hook and the reviewed GEMM `owned_container_v4.py` (SHA53ee4270…). Original raw
+archive hashes come from that pinned accepted manifest; author did not scan or
+load raw states, model payloads or large DSOs.
+
+Both source profiles remain separate: `official-bf16-loader` then
+`source-36-f32-restored`, each teacher11 then353 with its own fresh cache. The
+original loader/casts, source/RECORD audits, parameter audit, module hooks,
+token calls, snapshot codecs and unchanged archive validator remain in use.
+`CAPTURE_INTEGRATION.review.diff` exposes every insertion. Original named
+functions are AST-identical except delegation at the beginning of the existing
+`observe_calls` callback. No replacement profiler, tensor, operator or model.
+
+## New observation and transparency gates
+
+- Fresh serialized plain then observed containers, each running the accepted
+  two-profile/two-step procedure. No historical failed attempt is reused.
+- Copied exact `AttentionProjectionObserver` observes six FullAttention layers
+  3/7/11/15/19/23, nine BF16 tensors each, four profile/steps:216 tensors total.
+  Stages: hidden, query/gate projection, K/V projections, raw query/key,
+  normalized query/key, gate. Actual original module hooks and source-return
+  gate copy only; original profiler call counts remain authoritative.
+- Expected genuine2B shapes: hidden/gate2048, query/gate projection4096,
+  query8×256, K/V2×256. Raw query stride512 preserves interleaved query/gate.
+  These are stage transport descriptors, not reconstructed expected arithmetic.
+- Every fresh mode must retain all196 raw tensors, four full248320 BF16 logits,
+ 168 canonical domains and60 old intermediates byte-identical to the accepted
+  reference. Each mode also passes the unchanged complete archive/layout
+  validator;632 source/617 loaded parameter audit bytes match exactly.
+- Stdlib comparator reads bounded Safetensors descriptors and exact payloads,
+  including signed zeros; no Torch import, rounding, tolerance or tensor math.
+  Source/profiles/teachers/all-layer stage namespace, finite raw BF16,
+  dtype/shape/strides/cumulative byte cap and all cleanup evidence are required.
+- All base and new hooks are owned inside one finally boundary, every removal
+  attempted, profiler retired, primary and cleanup failures retained. Observer
+  module references are released before model disposal/archive validation.
+
+## Bounds and lifetime
+
+- Existing imageIDf57d77f5…/codec manifest eec72f47…; exact local inspect, no pull.
+- No NVIDIA CDI/device, CUDA_VISIBLE_DEVICES=-1, NVIDIA_VISIBLE_DEVICES=void,
+  no preload/audit, networknone, read-only source/probe/rootfs, cap-dropall.
+- Memory+swap12GiB, CPUquota4, original4 Torch threads/interops1, AVX2. Affinity
+  sets exactly four allowed CPUs before Torch import and validates afterward;
+  no unavailable OCI cpuset controller is requested.
+- Each capture300s outer/290s original alarm; original aggregate capture256MiB,
+  per-file256MiB, helper log4MiB. Additional attention raw payload172032B/step,
+  688128B total; its independent cumulative cap64MiB remains unchanged.
+- Original4.548GB checkpoint/model memory estimate remains an estimate, not
+  physical admission. Actual cgroup peak/RSS remain in each reference manifest.
+- Reviewed helper is copied unchanged: WNOWAIT keeps owned PID reserved;
+  independent TERM/grace/two KILL stages precede reap, exact nonce/name/CID
+  discovery before and after, group/container absence and pipe EOF/closure.
+  Postspawn receipt errors cannot bypass cleanup. Failed evidence prevents the
+  next mode and final PASS. Existing processes/containers are never targeted.
+- Final bounded stdlib comparison120s, block≤1MiB, tensor/archive geometry and
+  complete EOF enforced. Separate fresh output directly beneath this run.
+
+## ROOT-only commands after source review
+
+```sh
+cd /home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020
+PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_validate_pair
+PYTHONDONTWRITEBYTECODE=1 python3 run_pair.py --output "$PWD/root-pair-1"
+```
+
+Ten fixture methods are prepared, not run by author. Synthetic bytes test strict
+raw comparison, missing stages, teacher/profile binding, query stride and finite
+guards; they are never genuine trained expectations. ROOT should independently
+authenticate all prepared source and parent pins before either command.
+
+Final possible scope: `PASS_CPU_HF2B_PAIRED_ATTENTION_OBSERVATION_ONLY`. This
+qualifies observation transparency and actual reference-stage availability for
+the named two CPU profiles/two teacher tokens. It does not qualify Rust private
+slice output, full HF execution/SessionAuthority/StateRoot, GPU/FLA, attention
+RoPE/BMM/softmax integration, long context, trained27B or matched performance.
--- /dev/null
+++ TODO.md
@@ -0,0 +1,13 @@
+# Own run — original HF2B paired attention observation
+
+- Parent: `.agents/plans/model-execution/TODO.md`; owner baseline_review.
+- Owned scope: this ignored preparation directory only; no shared tracker.
+- [x] Read accepted two-profile CPU capture/validator/root-probe-2 metadata and Batch frozen hook.
+- [x] Preserve exact original source/parameter/semantic/raw math path; finally-owned observation integration.
+- [x] Prepare exact reviewed owned-container lifecycle, serialized fresh plain/observed launcher and strict raw comparator.
+- [x] Prepare216 actual-stage contract and10 source-free fixtures; retain explicit CPU-only/reference-only labels.
+- [x] AST/static source checks and frozen source/parent pins only; no executed tests or model/runtime claims.
+- [ ] Independent source review and ROOT fixture execution.
+- [ ] ROOT paired capture; each mode all196 raw tensors/full4 logits/all168 domains/60 intermediates byte-exact to accepted reference.
+- [ ] ROOT actual216 attention stages, source audit, cgroup/affinity and full owned cleanup validation.
+- [ ] Later separate Rust operator-slice replay against these authentic stages; full HF/session/GPU guards remain closed.
--- /dev/null
+++ attention_observation_hook.py
@@ -0,0 +1,189 @@
+"""ROOT ONLY. Transparent hooks; outer accepted loader/source/lifetime stay owned.
+
+No model factory or token loop. Integrate observe_source_return into the existing
+accepted profiler callback; never replace it or set a new global profiler here.
+Full logits/cache transparency remains an independent outer acceptance gate.
+"""
+import hashlib
+import json
+from pathlib import Path
+
+MODELING_SHA256 = "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
+TORCH_SOURCE = "cf30153c4c131c8164ee7798e5022d810682e2cb"
+PROFILES = {"official-bf16-loader", "source-36-f32-restored"}
+MAX_PAYLOAD = 64 * 1024 * 1024
+STAGES = ("hidden", "query_gate_projection", "key_projection", "value_projection",
+          "query", "key", "normalized_query", "normalized_key", "gate")
+
+
+def require(value, message):
+    if not value:
+        raise RuntimeError(message)
+
+
+class AttentionProjectionObserver:
+    """Observe genuine CPU public modules at all actual FullAttention layers.
+
+    Call only after the unchanged accepted loader and parameter/source audit.
+    No hook returns a replacement. The copy precedes possible downstream reuse.
+    """
+
+    def __init__(self, model, official):
+        import torch
+        self.torch = torch
+        self.official = official
+        self.source_path = str(Path(official.__file__).resolve())
+        require(hashlib.sha256(Path(self.source_path).read_bytes()).hexdigest()
+                == MODELING_SHA256, "exact accepted original modeling source")
+        require(torch.version.git_version == TORCH_SOURCE,
+                "exact accepted Torch source")
+        require(torch.backends.cpu.get_cpu_capability() == "AVX2",
+                "closed original CPU runtime capability")
+        self.language = model.model.language_model
+        cfg = model.config.text_config
+        require(cfg.hidden_size in (1024, 2048) and cfg.head_dim == 256,
+                "approved .8B/2B attention observation geometry")
+        self.hidden = cfg.hidden_size
+        self.query_heads = cfg.num_attention_heads
+        self.kv_heads = cfg.num_key_value_heads
+        self.layers = {}
+        for index, layer in enumerate(self.language.layers):
+            if cfg.layer_types[index] != "full_attention":
+                continue
+            attention = layer.self_attn
+            require(isinstance(attention, official.Qwen3_5Attention)
+                    and attention.layer_idx == index, "actual original attention module")
+            require(tuple(attention.q_proj.weight.shape)
+                    == (2*self.query_heads*256, self.hidden)
+                    and tuple(attention.k_proj.weight.shape)
+                    == tuple(attention.v_proj.weight.shape)
+                    == (self.kv_heads*256, self.hidden), "unchanged full source matrices")
+            require(all(parameter.device.type == "cpu" for parameter in attention.parameters()),
+                    "CPU-only actual source module")
+            self.layers[index] = attention
+        require(len(self.layers) == 6, "all six original FullAttention modules")
+        self.handles = []
+        self.active = None
+        self.seen = set()
+        self.payload = 0
+        self.tensors = {}
+        self.metadata = {}
+        try:
+            for index, attention in self.layers.items():
+                self.handles.append(attention.register_forward_pre_hook(
+                    self._input_hook(index, "hidden"), with_kwargs=True))
+                for member, name in (("q_proj", "query_gate_projection"),
+                                     ("k_proj", "key_projection"),
+                                     ("v_proj", "value_projection"),
+                                     ("q_norm", "normalized_query"),
+                                     ("k_norm", "normalized_key")):
+                    self.handles.append(getattr(attention, member).register_forward_hook(
+                        self._output_hook(index, name)))
+                for member, name in (("q_norm", "query"), ("k_norm", "key")):
+                    self.handles.append(getattr(attention, member).register_forward_pre_hook(
+                        self._input_hook(index, name), with_kwargs=True))
+        except BaseException as primary:
+            try:
+                self.close()
+            except BaseException as cleanup:
+                raise BaseExceptionGroup("observer install and removal failed", [primary, cleanup]) from None
+            raise
+
+    def _remember(self, layer, stage, value):
+        torch = self.torch
+        require(self.active is not None, "unexpected attention call outside fixed teacher step")
+        key = f"layer-{layer:02d}.{stage}"
+        require(key not in self.tensors, "duplicate genuine attention stage: " + key)
+        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
+                and value.dtype == torch.bfloat16 and value.numel() <= 16384,
+                "closed actual BF16 attention tensor: " + key)
+        require(torch.is_inference_mode_enabled(), "unchanged accepted inference mode")
+        expected = {"hidden": self.hidden,
+                    "query_gate_projection": 2*self.query_heads*256,
+                    "key_projection": self.kv_heads*256,
+                    "value_projection": self.kv_heads*256,
+                    "query": self.query_heads*256, "key": self.kv_heads*256,
+                    "normalized_query": self.query_heads*256,
+                    "normalized_key": self.kv_heads*256, "gate": self.query_heads*256}[stage]
+        require(value.numel() == expected and value.shape[0] == value.shape[1] == 1,
+                "single teacher original attention shape: " + key)
+        count = value.numel()*value.element_size()
+        require(self.payload+count <= MAX_PAYLOAD, "bounded attention-only tensor payload")
+        copied = value.detach().cpu().contiguous().clone()
+        require(torch.equal(value.view(torch.uint16), copied.view(torch.uint16)),
+                "bit-preserving observational copy, including signed zero")
+        self.payload += count
+        self.tensors[key] = copied
+        self.metadata[key] = {"dtype": str(value.dtype), "shape": list(value.shape),
+                              "original_stride": list(value.stride()),
+                              "saved_stride": list(copied.stride()), "bytes": count,
+                              "source_module_layer": layer, "stage": stage}
+        # Implicit None: never return a replacement to a public module hook.
+
+    def _input_hook(self, layer, stage):
+        def observe(_module, args, kwargs):
+            require(bool(args) != ("hidden_states" in kwargs),
+                    "one genuine public activation argument")
+            self._remember(layer, stage, args[0] if args else kwargs["hidden_states"])
+        return observe
+
+    def _output_hook(self, layer, stage):
+        def observe(_module, _args, output):
+            self._remember(layer, stage, output)
+        return observe
+
+    def begin_step(self, profile, step, token):
+        require(self.active is None and not self.tensors, "previous attention step retired")
+        require(profile in PROFILES and step in (0, 1) and token == (11, 353)[step],
+                "same accepted two-token teacher protocol")
+        require((profile, step) not in self.seen, "no repeated source-profile step")
+        self.seen.add((profile, step))
+        self.active = {"profile": profile, "step": step, "teacher_token": token}
+
+    def observe_source_return(self, frame, event, _arg):
+        # Delegate from the existing accepted callback, without changing its
+        # original GDN/conv/attention call-count observations or source math.
+        if event != "return" or frame.f_code.co_filename != self.source_path:
+            return
+        owner = frame.f_locals.get("self")
+        if frame.f_code.co_name != "forward" or not isinstance(owner, self.official.Qwen3_5Attention):
+            return
+        layer = owner.layer_idx
+        require(self.layers.get(layer) is owner, "genuine observed original layer owner")
+        require("gate" in frame.f_locals, "actual untouched forward gate local")
+        self._remember(layer, "gate", frame.f_locals["gate"])
+
+    def finish_step(self, directory):
+        from safetensors.torch import save_file
+        expected = {f"layer-{layer:02d}.{stage}" for layer in self.layers for stage in STAGES}
+        require(self.active is not None and set(self.tensors) == expected,
+                "all actual layers/stages exactly once")
+        directory = Path(directory)
+        directory.mkdir(exist_ok=False)
+        archive = directory/"attention.safetensors"
+        save_file(self.tensors, str(archive), metadata={
+            "scope": "actual public source observation only; full transparency unqualified",
+            **{key: str(value) for key, value in self.active.items()}})
+        report = {"status": "OBSERVED_OPERATORS_ONLY_FULL_TRANSPARENCY_UNQUALIFIED",
+                  **self.active, "modeling_sha256": MODELING_SHA256,
+                  "torch_source": TORCH_SOURCE, "tensor_count": len(self.tensors),
+                  "aggregate_payload_bytes": self.payload, "tensors": self.metadata,
+                  "archive": {"file": archive.name, "bytes": archive.stat().st_size,
+                              "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()},
+                  "no_model_or_state_or_provider_eligibility": True}
+        (directory/"observation.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
+        self.tensors = {}
+        self.metadata = {}
+        self.active = None
+        return report
+
+    def close(self):
+        errors = []
+        for handle in reversed(self.handles):
+            try:
+                handle.remove()
+            except BaseException as error:
+                errors.append(repr(error))
+        self.handles = []
+        if errors:
+            raise RuntimeError("all owned hook removals attempted: " + repr(errors))
--- /dev/null
+++ baseline/capture_reference.py
@@ -0,0 +1,373 @@
+"""Pinned, offline CPU reference; operators remain the installed official source."""
+
+import collections
+import gc
+import signal
+import stat
+import hashlib
+import importlib.metadata
+import inspect
+import json
+import os
+import pathlib
+import platform
+import resource
+import sys
+import time
+
+import torch
+from safetensors import safe_open
+from safetensors.torch import save_file
+from transformers import AutoTokenizer
+from transformers.models.qwen3_5 import modeling_qwen3_5 as official
+
+SOURCE = pathlib.Path("/source")
+PINS = json.loads(pathlib.Path("/probe/source-pins.json").read_text())
+assert PINS["repo"] == "Qwen/Qwen3.5-2B" and PINS["revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
+AUDIT_ELEMENTS = 262144  # ≤1MiB per F32 audit operand; model operators untouched.
+ARTIFACT_CAP = 256 * 1024 * 1024
+signal.alarm(290)  # Outer owner enforces300s and always cleans its bound CID.
+resource.setrlimit(resource.RLIMIT_FSIZE, (ARTIFACT_CAP, ARTIFACT_CAP))
+assert pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip() == str(12 * 1024**3)
+assert pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip() == "400000 100000"
+assert os.environ["ATEN_CPU_CAPABILITY"] == "avx2"
+OUTPUT = pathlib.Path("/output/captured-reference")
+TOKENS = [11, 353]
+PROFILES = ["official-bf16-loader", "source-36-f32-restored"]
+torch.set_num_threads(4)
+torch.set_num_interop_threads(1)
+assert not torch.cuda.is_available()
+assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
+assert not OUTPUT.exists(), "refuse to overwrite an earlier reference"
+OUTPUT.mkdir()
+
+
+def sha(path):
+    digest = hashlib.sha256()
+    with open(path, "rb") as stream:
+        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
+            digest.update(chunk)
+    return digest.hexdigest()
+
+
+def write_json(path, data):
+    def official_json_types(value):
+        if isinstance(value, set):
+            return sorted(value)
+        raise TypeError(f"unhandled JSON type {type(value).__name__}")
+
+    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False, default=official_json_types) + "\n")
+
+
+def verify_source():
+    for item in PINS["source_files"]:
+        path = SOURCE / item["path"]
+        assert not path.is_symlink() and path.is_file() and path.stat().st_size == item["bytes"]
+        assert sha(path) == item["sha256"], item["path"]
+    return {"repo": PINS["repo"], "revision": PINS["revision"], "snapshot": PINS["snapshot"],
+            "files": PINS["source_files"], "independent_verification": PINS["root_source_validation"]}
+
+
+def audit_chunks(value):
+    assert value.device.type == "cpu" and value.is_contiguous()
+    flat = value.detach().view(-1)
+    for start in range(0, flat.numel(), AUDIT_ELEMENTS):
+        yield flat[start:start + AUDIT_ELEMENTS]
+
+
+def finite(value):
+    return all(bool(torch.isfinite(part).all()) for part in audit_chunks(value.detach().contiguous()))
+
+
+def artifact_bound():
+    total = 0
+    for path in OUTPUT.rglob("*"):
+        assert not path.is_symlink()
+        if path.is_file():
+            total += path.stat().st_size
+    assert total <= ARTIFACT_CAP, ("aggregate capture cap", total)
+    return total
+
+
+def verify_runtime():
+    expected = PINS["runtime_source"]
+    for part in expected["files"].values():
+        path = pathlib.Path(part["runtime_path"])
+        assert sha(path) == part["sha256"], str(path)
+    for name, pin in PINS["package_records"].items():
+        distribution = importlib.metadata.distribution(name)
+        assert distribution.version == pin["version"], name
+        record = distribution.read_text("RECORD")
+        assert record is not None and hashlib.sha256(record.encode()).hexdigest() == pin["record_sha256"], name
+    assert torch.__version__ == "2.13.0+cu129" and importlib.metadata.version("transformers") == "5.17.0"
+    assert torch.backends.cpu.get_cpu_capability() == "AVX2"
+    return expected
+
+
+def tensor_info(value):
+    return {
+        "dtype": str(value.dtype), "shape": list(value.shape),
+        "stride": list(value.stride()), "numel": value.numel(),
+        "bytes": value.numel() * value.element_size(),
+        "device": str(value.device), "contiguous": value.is_contiguous(),
+        "finite": finite(value),
+    }
+
+
+def f32_file(directory, name, value):
+    path = directory / (name + ".f32le.bin")
+    wide = value.detach().cpu().float().contiguous()
+    assert finite(wide)
+    # numpy explicitly requests little-endian; widening BF16 is exact.
+    path.write_bytes(wide.numpy().astype("<f4", copy=False).tobytes())
+    return {"file": str(path.relative_to(OUTPUT)), "sha256": sha(path), **tensor_info(wide)}
+
+
+def parameter_audit(model, source_file):
+    rows = []
+    with safe_open(source_file, framework="pt", device="cpu") as handle:
+        for name in sorted(handle.keys()):
+            original = handle.get_tensor(name)
+            try:
+                actual = model.get_parameter(name)
+            except AttributeError:
+                assert name.startswith("mtp."), name
+                rows.append({"name": name, "source": tensor_info(original), "loaded": None, "exact_after_declared_dtype_cast": None, "exclusion": "official Qwen3_5PreTrainedModel._keys_to_ignore_on_load_unexpected=['^mtp.*']; auxiliary MTP is absent from the ordinary conditional-generation class"})
+                continue
+            assert actual.device.type == "cpu"
+            assert actual.shape == original.shape
+            assert original.is_contiguous() and actual.is_contiguous()
+            cast_changed, cast_max = 0, 0.0
+            for left, right in zip(audit_chunks(original), audit_chunks(actual)):
+                cast = left.to(right.dtype)
+                assert torch.equal(cast.view(torch.uint8), right.view(torch.uint8)), name
+                if original.dtype != actual.dtype:
+                    delta = left.float() - right.float()
+                    cast_changed += int((delta != 0).sum())
+                    cast_max = max(cast_max, float(delta.abs().max()))
+            row = {"name": name, "source": tensor_info(original), "loaded": tensor_info(actual)}
+            row["exact_after_declared_dtype_cast"] = True
+            if original.dtype != actual.dtype:
+                row["cast_changed_elements"] = cast_changed
+                row["cast_max_abs_delta"] = cast_max
+            rows.append(row)
+    return rows
+
+
+def snapshot(cache, logits, directory, profile, step, token):
+    directory.mkdir()
+    raw = {"logits": logits.detach().cpu().contiguous().clone()}
+    raw_meta = {"logits": tensor_info(logits)}
+    domains = []
+    assert len(cache.layers) == 24
+    assert logits.shape == (1, 1, 248320)
+    for index, layer in enumerate(cache.layers):
+        prefix = f"layer-{index:02d}"
+        if hasattr(layer, "conv_states"):
+            assert set(layer.conv_states) == {0} and set(layer.recurrent_states) == {0}
+            conv, recurrent = layer.conv_states[0], layer.recurrent_states[0]
+            assert conv.shape == (1, 6144, 4)
+            assert recurrent.shape == (1, 16, 128, 128)
+            assert recurrent.dtype == torch.float32 and conv.dtype == torch.bfloat16
+            for kind, value in [("conv", conv), ("recurrent", recurrent)]:
+                raw[prefix + "." + kind] = value.detach().cpu().contiguous().clone()
+                raw_meta[prefix + "." + kind] = tensor_info(value)
+            # Official cache retains four raw projection positions. The next
+            # four-tap convolution needs the newest three as prior history.
+            domains.append({"layer": index, "kind": "gdn-convolution-history", "layout": "batch,channel,oldest-to-newest-3", "derivation": "official conv_states[0][..., -3:]", **f32_file(directory, prefix + ".conv-history", conv[..., -3:])})
+            # Source state is key-major; the Rust state reader is value-major.
+            domains.append({"layer": index, "kind": "gdn-recurrent", "layout": "batch,head,value,key", "derivation": "official recurrent_states[0].permute(0,1,3,2)", **f32_file(directory, prefix + ".recurrent-value-major", recurrent.permute(0, 1, 3, 2))})
+        else:
+            keys, values = layer.keys, layer.values
+            assert keys.shape == values.shape == (1, 2, step + 1, 256)
+            assert keys.dtype == values.dtype == torch.bfloat16
+            for kind, value in [("keys", keys), ("values", values)]:
+                raw[prefix + "." + kind] = value.detach().cpu().contiguous().clone()
+                raw_meta[prefix + "." + kind] = tensor_info(value)
+            keys = keys.permute(0, 2, 1, 3).contiguous()
+            values = values.permute(0, 2, 1, 3).contiguous()
+            pair = torch.cat([keys.reshape(-1), values.reshape(-1)])
+            domains.append({"layer": index, "kind": "attention-kv-pair", "layout": "all keys(batch,position,kv-head,width), then all values(same)", "valid_positions": step + 1, "derivation": "official keys/values permute(0,2,1,3), flatten and concatenate", **f32_file(directory, prefix + ".kv-pair", pair)})
+    assert len(domains) == 42
+    assert len(raw) == 49  # logits + 36 GDN tensors + 12 K/V tensors
+    assert all(info["finite"] for info in raw_meta.values())
+    raw_path = directory / "raw-official.safetensors"
+    save_file(raw, str(raw_path), metadata={"profile": profile, "step": str(step), "input_token": str(token), "source_revision": PINS["revision"]})
+    wide_logits = f32_file(directory, "logits", logits)
+    flat = logits.float().reshape(-1)
+    top_values, top_ids = torch.topk(flat, 10, largest=True, sorted=True)
+    result = {"profile": profile, "step": step, "input_token": token, "consumed_tokens": TOKENS[:step + 1], "cache_class": type(cache).__name__, "cache_seq_length": cache.get_seq_length(), "raw_archive": {"file": str(raw_path.relative_to(OUTPUT)), "sha256": sha(raw_path), "bytes": raw_path.stat().st_size}, "raw_tensor_layouts": raw_meta, "logits_f32": wide_logits, "argmax_token_id": int(flat.argmax()), "top10": [{"token_id": int(i), "logit": float(v)} for i, v in zip(top_ids, top_values)], "semantic_domains": domains, "semantic_domain_count": len(domains), "physical_ready_or_StateRoot_claim": False}
+    write_json(directory / "manifest.json", result)
+    return result
+
+
+started = time.time()
+runtime_source = verify_runtime()
+source_receipt = verify_source()
+write_json(OUTPUT / "source-receipt.json", source_receipt)
+source_file = SOURCE / "model.safetensors-00001-of-00001.safetensors"
+with safe_open(source_file, framework="pt", device="cpu") as source_handle:
+    names = sorted(source_handle.keys())
+    f32_names = [name for name in names if source_handle.get_slice(name).get_dtype() == "F32"]
+assert len(names) == 632 and len(f32_names) == 36
+assert f32_names == PINS["raw_f32_names"]
+assert all(".linear_attn.A_log" in name or ".linear_attn.norm.weight" in name for name in f32_names)
+assert "lm_head.weight" not in names
+print("SOURCE_VERIFIED tensors=632 original_f32=36 no_lm_head=1", flush=True)
+
+model, loading_info = official.Qwen3_5ForConditionalGeneration.from_pretrained(
+    str(SOURCE), dtype=torch.bfloat16, local_files_only=True,
+    attn_implementation="eager", use_kernels=False, output_loading_info=True,
+)
+model.eval()
+assert not loading_info.get("missing_keys") and not loading_info.get("unexpected_keys") and not loading_info.get("mismatched_keys") and not loading_info.get("error_msgs"), loading_info
+assert all(parameter.device.type == "cpu" for parameter in model.parameters())
+embedding = model.model.language_model.embed_tokens.weight
+assert model.lm_head.weight is embedding
+assert embedding.shape == (248320, 2048)
+assert model.config.text_config.hidden_size == 2048 and model.config.text_config.num_hidden_layers == 24
+tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True)
+print("OFFICIAL_MODEL_LOADED tied_head=1 cuda=0", flush=True)
+
+# Observation hooks copy actual outputs, return None and never replace an
+# operator/input/output. The down-projection input exposes the actual fused
+# SiLU(gate)*up tensor without reconstructing its arithmetic.
+intermediates = {}
+handles = []
+
+
+def remember(name, value):
+    assert name not in intermediates, name
+    assert isinstance(value, torch.Tensor)
+    intermediates[name] = value.detach().cpu().contiguous().clone()
+
+
+def output_hook(module, name):
+    handles.append(module.register_forward_hook(lambda _module, _args, value: remember(name, value)))
+
+
+def input_hook(module, name):
+    handles.append(module.register_forward_pre_hook(lambda _module, args: remember(name, args[0])))
+
+
+language = model.model.language_model
+layer0 = language.layers[0]
+output_hook(language.embed_tokens, "embedding")
+output_hook(layer0.input_layernorm, "layer-00.attention_input_norm")
+output_hook(layer0.linear_attn.in_proj_qkv, "layer-00.gdn_qkv")
+output_hook(layer0.linear_attn, "layer-00.attention_output")
+input_hook(layer0.post_attention_layernorm, "layer-00.post_attention_residual")
+output_hook(layer0.post_attention_layernorm, "layer-00.ffn_input_norm")
+output_hook(layer0.mlp.gate_proj, "layer-00.ffn_gate")
+output_hook(layer0.mlp.up_proj, "layer-00.ffn_up")
+input_hook(layer0.mlp.down_proj, "layer-00.ffn_fused")
+output_hook(layer0.mlp.down_proj, "layer-00.ffn_output")
+output_hook(layer0, "layer-00.post_ffn_residual")
+output_hook(language.norm, "final_norm")
+
+manifest = {
+    "schema": "zerogravity.official-cpu-qwen35-reference.v2-hf2b",
+    "source_pins_sha256": sha(pathlib.Path("/probe/source-pins.json")),
+    "source_repo": source_receipt["repo"], "source_revision": source_receipt["revision"],
+    "image": PINS["derived_image_id"], "base_image": PINS["base_image"],
+    "runtime_source": runtime_source, "package_records": PINS["package_records"],
+    "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "safetensors": importlib.metadata.version("safetensors"), "vllm": importlib.metadata.version("vllm"), "torch_num_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "cuda_available": torch.cuda.is_available(), "mkldnn_enabled": torch.backends.mkldnn.enabled, "default_dtype": str(torch.get_default_dtype()), "cpu_capability": torch.backends.cpu.get_cpu_capability(), "machine": platform.machine(), "processor": platform.processor()},
+    "loader": {"model_class": type(model).__name__, "dtype": "torch.bfloat16", "attn_implementation": "eager", "use_kernels": False, "local_files_only": True, "loading_info": loading_info},
+    "config": model.config.to_dict(), "tokenizer": {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer), "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id, "tokens": TOKENS, "token_strings": tokenizer.convert_ids_to_tokens(TOKENS), "decoded": tokenizer.decode(TOKENS, skip_special_tokens=False)},
+    "head": {"tied_parameter_object": True, "source_has_lm_head_tensor": False, "embedding_name": "model.language_model.embed_tokens.weight", "output_dtype_rule": "official lm_head(hidden); no loss, no automatic FP32 upcast"},
+    "protocol": "one cached official forward per fixed input token, no sampling or generation; first token uses chunk reference, next token recurrent reference",
+    "limits": {"container_memory_max": pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip(), "container_cpu_max": pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip(), "network": "none", "source_mount": "read-only", "gpu_exposed": False},
+    "scope": "Official Transformers eager CPU attention and slow PyTorch GDN operators only. No GPU FLA, vLLM inference, throughput, quantization, Rust bit parity, or trained27B qualification.",
+    "profiles": {},
+    "source_geometry": {"hidden": 2048, "ffn": 6144, "layers": 24, "gdn_layers": 18, "full_attention_layers": 6, "gdn_heads": 16, "gdn_head_width": 128, "kv_heads": 2, "kv_head_width": 256},
+}
+model_source = pathlib.Path(inspect.getsourcefile(official))
+manifest["official_modeling_source"] = {"path": str(model_source), "sha256": sha(model_source)}
+manifest["capture_script_sha256"] = sha(pathlib.Path(__file__))
+
+for profile in PROFILES:
+    if profile == "source-36-f32-restored":
+        with safe_open(source_file, framework="pt", device="cpu") as source_handle:
+            for name in f32_names:
+                parameter = model.get_parameter(name)
+                parameter.data = source_handle.get_tensor(name).clone()
+                assert parameter.dtype == torch.float32
+        print("RESTORED_ORIGINAL_F32_PARAMETERS count=36", flush=True)
+    profile_dir = OUTPUT / profile
+    profile_dir.mkdir()
+    audit = parameter_audit(model, source_file)
+    write_json(profile_dir / "parameter-audit.json", audit)
+    assert len(audit) == 632 and sum(row["loaded"] is None for row in audit) == 15
+    dtype_counts = collections.Counter(row["loaded"]["dtype"] for row in audit if row["loaded"] is not None)
+    print("PROFILE_AUDIT", profile, dict(dtype_counts), flush=True)
+    profile_result = {"profile": profile, "parameter_audit_sha256": sha(profile_dir / "parameter-audit.json"), "source_parameter_count": len(audit), "actual_checkpoint_parameter_dtype_counts": dict(dtype_counts), "source_f32_parameter_names": f32_names, "restoration": "none; unmodified official requested-BF16 loader" if profile == PROFILES[0] else "only the 36 originally F32 checkpoint parameters restored to their exact source dtype and bytes; official operators unchanged", "steps": []}
+    cache = None
+    with torch.inference_mode():
+        for step, token in enumerate(TOKENS):
+            intermediates = {}
+            calls = collections.Counter()
+            signatures = {}
+
+            def observe_calls(frame, event, arg):
+                if frame.f_code.co_filename != str(model_source):
+                    return
+                name = frame.f_code.co_name
+                if event == "return" and name in {"causal_conv1d_fn", "causal_conv1d_update"}:
+                    caller = frame.f_back
+                    while caller is not None:
+                        owner = caller.f_locals.get("self")
+                        if isinstance(owner, official.Qwen3_5GatedDeltaNet):
+                            if owner.layer_idx == 0:
+                                remember("layer-00.gdn_conv_silu", arg.transpose(1, 2))
+                            break
+                        caller = caller.f_back
+                    return
+                if event != "call":
+                    return
+                if name in {"torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update", "eager_attention_forward"}:
+                    key = name + ":" + str(frame.f_code.co_firstlineno)
+                    calls[key] += 1
+                    signatures[key] = {key: tensor_info(value) for key, value in frame.f_locals.items() if isinstance(value, torch.Tensor)}
+
+            step_start = time.time()
+            sys.setprofile(observe_calls)
+            try:
+                result = model(input_ids=torch.tensor([[token]], dtype=torch.int64), past_key_values=cache, use_cache=True, return_dict=True)
+            finally:
+                sys.setprofile(None)
+            cache = result.past_key_values
+            expected = {("torch_chunk_gated_delta_rule" if step == 0 else "torch_recurrent_gated_delta_rule"): 18, ("causal_conv1d_fn" if step == 0 else "causal_conv1d_update"): 18, "eager_attention_forward": 6}
+            observed_by_name = {key.split(":")[0]: value for key, value in calls.items()}
+            assert observed_by_name == expected, (expected, observed_by_name)
+            step_result = snapshot(cache, result.logits, profile_dir / f"step-{step}", profile, step, token)
+            remember("final_logits", result.logits)
+            remember("layer-00.gdn_recurrent_fp32", cache.layers[0].recurrent_states[0].permute(0, 1, 3, 2))
+            assert len(intermediates) == 15
+            intermediate_path = profile_dir / f"step-{step}" / "observed-intermediates.safetensors"
+            save_file(intermediates, str(intermediate_path), metadata={"profile": profile, "step": str(step), "scope": "actual layer0/module input/output copies; no reconstructed arithmetic"})
+            step_result["observed_intermediates"] = {"file": str(intermediate_path.relative_to(OUTPUT)), "sha256": sha(intermediate_path), "bytes": intermediate_path.stat().st_size, "tensors": {name: tensor_info(value) for name, value in intermediates.items()}, "hook_scope": "official module pre/post hooks, CPU tensor clone only; slow causal-conv source return observed with sys.setprofile; no replacement return"}
+            step_result["observed_official_operator_calls"] = dict(calls)
+            step_result["last_observed_operator_tensor_arguments"] = signatures
+            step_result["diagnostic_elapsed_seconds_not_benchmark"] = time.time() - step_start
+            write_json(profile_dir / f"step-{step}" / "manifest.json", step_result)
+            profile_result["steps"].append(step_result)
+            artifact_bound()
+            print("REFERENCE_STEP_COMPLETE", profile, "step", step, "argmax", step_result["argmax_token_id"], "domains", len(step_result["semantic_domains"]), "seconds", step_result["diagnostic_elapsed_seconds_not_benchmark"], flush=True)
+    manifest["profiles"][profile] = profile_result
+    write_json(OUTPUT / "manifest.json", manifest)
+    del result, cache
+
+for handle in handles:
+    handle.remove()
+del model, embedding, language, layer0, intermediates
+gc.collect()
+assert verify_source() == source_receipt
+assert verify_runtime() == runtime_source
+manifest["artifact_bytes_before_final_manifest"] = artifact_bound()
+manifest["cgroup_memory_peak_bytes"] = pathlib.Path("/sys/fs/cgroup/memory.peak").read_text().strip()
+manifest["source_reverified_after_capture"] = True
+manifest["elapsed_seconds_not_benchmark"] = time.time() - started
+manifest["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
+write_json(OUTPUT / "manifest.json", manifest)
+artifact_bound()
+print("REFERENCE_CAPTURE_COMPLETE", json.dumps({"profiles": list(manifest["profiles"]), "steps_each": 2, "full_vocab": 248320, "semantic_domains_each": 42, "max_rss_kib": manifest["max_rss_kib"]}), flush=True)
--- /dev/null
+++ baseline/source-pins.json
@@ -0,0 +1,226 @@
+{
+  "schema": "zerogravity.hf2b-reference-pins.v1",
+  "repo": "Qwen/Qwen3.5-2B",
+  "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
+  "snapshot": "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52",
+  "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52/files",
+  "source_files": [
+    {
+      "path": "LICENSE",
+      "bytes": 11544,
+      "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "README.md",
+      "bytes": 62814,
+      "sha256": "c0e83a849c776e6fa843d011f023132d942a0f0140d903205bf1c363adad2275",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "chat_template.jinja",
+      "bytes": 7755,
+      "sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "config.json",
+      "bytes": 2908,
+      "sha256": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "merges.txt",
+      "bytes": 3353259,
+      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "model.safetensors-00001-of-00001.safetensors",
+      "bytes": 4548221488,
+      "sha256": "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
+      "provenance": "independent official LFS SHA256"
+    },
+    {
+      "path": "model.safetensors.index.json",
+      "bytes": 64460,
+      "sha256": "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "preprocessor_config.json",
+      "bytes": 390,
+      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "tokenizer.json",
+      "bytes": 12807982,
+      "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
+      "provenance": "independent official LFS SHA256"
+    },
+    {
+      "path": "tokenizer_config.json",
+      "bytes": 16709,
+      "sha256": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "video_preprocessor_config.json",
+      "bytes": 385,
+      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "vocab.json",
+      "bytes": 6722759,
+      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    }
+  ],
+  "raw_f32_names": [
+    "model.language_model.layers.0.linear_attn.A_log",
+    "model.language_model.layers.0.linear_attn.norm.weight",
+    "model.language_model.layers.1.linear_attn.A_log",
+    "model.language_model.layers.1.linear_attn.norm.weight",
+    "model.language_model.layers.10.linear_attn.A_log",
+    "model.language_model.layers.10.linear_attn.norm.weight",
+    "model.language_model.layers.12.linear_attn.A_log",
+    "model.language_model.layers.12.linear_attn.norm.weight",
+    "model.language_model.layers.13.linear_attn.A_log",
+    "model.language_model.layers.13.linear_attn.norm.weight",
+    "model.language_model.layers.14.linear_attn.A_log",
+    "model.language_model.layers.14.linear_attn.norm.weight",
+    "model.language_model.layers.16.linear_attn.A_log",
+    "model.language_model.layers.16.linear_attn.norm.weight",
+    "model.language_model.layers.17.linear_attn.A_log",
+    "model.language_model.layers.17.linear_attn.norm.weight",
+    "model.language_model.layers.18.linear_attn.A_log",
+    "model.language_model.layers.18.linear_attn.norm.weight",
+    "model.language_model.layers.2.linear_attn.A_log",
+    "model.language_model.layers.2.linear_attn.norm.weight",
+    "model.language_model.layers.20.linear_attn.A_log",
+    "model.language_model.layers.20.linear_attn.norm.weight",
+    "model.language_model.layers.21.linear_attn.A_log",
+    "model.language_model.layers.21.linear_attn.norm.weight",
+    "model.language_model.layers.22.linear_attn.A_log",
+    "model.language_model.layers.22.linear_attn.norm.weight",
+    "model.language_model.layers.4.linear_attn.A_log",
+    "model.language_model.layers.4.linear_attn.norm.weight",
+    "model.language_model.layers.5.linear_attn.A_log",
+    "model.language_model.layers.5.linear_attn.norm.weight",
+    "model.language_model.layers.6.linear_attn.A_log",
+    "model.language_model.layers.6.linear_attn.norm.weight",
+    "model.language_model.layers.8.linear_attn.A_log",
+    "model.language_model.layers.8.linear_attn.norm.weight",
+    "model.language_model.layers.9.linear_attn.A_log",
+    "model.language_model.layers.9.linear_attn.norm.weight"
+  ],
+  "tensor_counts": {
+    "F32": 36,
+    "BF16": 596
+  },
+  "dtype_elements": {
+    "F32": 2592,
+    "BF16": 2274067232
+  },
+  "root_source_validation": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-source-plan-20261020/root-post-acquisition-validation.json",
+    "bytes": 5496,
+    "sha256": "57929dd60414c1f2b4df536c05f39bcbef9957221ae45ae42f7982d6d1666ccb"
+  },
+  "cache_manifest": {
+    "path": "/home/pvl/.cache/abird/zerogravity/models/snapshots/88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52/manifest.json",
+    "bytes": 2246,
+    "sha256": "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52"
+  },
+  "base_capture": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/capture_reference.py",
+    "bytes": 18717,
+    "sha256": "225d8637f927f8bc6ce99af9d9dda6a6a5cd33fc9d7d24042dda7b98297f335d"
+  },
+  "base_validator": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/validate_reference.py",
+    "bytes": 5573,
+    "sha256": "5ef1077ed6abdb37093701cb2846fc044a35f62eb9d883891876c5e98f0ced09"
+  },
+  "runtime_source_manifest": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/runtime-source-manifest.json",
+    "bytes": 1942,
+    "sha256": "24eabe68b10c2ac6a17e2dcdcd53805abe47556a98b8b6a4f2f1c27c60e6c673"
+  },
+  "runtime_source": {
+    "python": "3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]",
+    "versions": {
+      "transformers": "5.17.0",
+      "torch": "2.13.0+cu129",
+      "safetensors": "0.8.0",
+      "vllm": "0.31.0+cu129",
+      "flash-linear-attention": null,
+      "fla-core": null,
+      "triton": "3.7.1"
+    },
+    "files": {
+      "transformers/models/qwen3_5/modeling_qwen3_5.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py",
+        "sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b",
+        "bytes": 95685,
+        "artifact": "pinned-runtime-source/modeling_qwen3_5.py"
+      },
+      "transformers/models/qwen3_5/configuration_qwen3_5.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_5/configuration_qwen3_5.py",
+        "sha256": "19966c3200cee92cc4bccaef5f94550c56d731dbf03858a9bd2ed6aebcc3f7da",
+        "bytes": 8395,
+        "artifact": "pinned-runtime-source/configuration_qwen3_5.py"
+      },
+      "transformers/cache_utils.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/cache_utils.py",
+        "sha256": "702144bb44553f6339ea1bf23c8205a708bb5f8c7c09cb3a2db484182646743c",
+        "bytes": 104677,
+        "artifact": "pinned-runtime-source/cache_utils.py"
+      },
+      "transformers/modeling_utils.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py",
+        "sha256": "38c2bd02ed7af229f54e2f02dae65663be7af29ed2c9498bd1c460cf60fbb62d",
+        "bytes": 268806,
+        "artifact": "pinned-runtime-source/modeling_utils.py"
+      },
+      "transformers/integrations/hub_kernels.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/integrations/hub_kernels.py",
+        "sha256": "c1eafefe0cbdf7f0deaa21b1ef6cb6d7c6082b3f486651d069ff38fcf3e53c95",
+        "bytes": 44974,
+        "artifact": "pinned-runtime-source/hub_kernels.py"
+      }
+    }
+  },
+  "derived_image_receipt": {
+    "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/torchcodec-cpu/derived-image-receipt.json",
+    "bytes": 7655,
+    "sha256": "c53c8421af4c832b901c00f4aabb65c7a1901b92c0655af188fa886fdaac63c0"
+  },
+  "derived_image_id": "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
+  "base_image": "docker.io/vllm/vllm-openai@sha256:b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f",
+  "package_records": {
+    "torch": {
+      "record_sha256": "4ae8e32005981a22c51a85130862a0ca173a01441a9de439b405a380c8dba243",
+      "version": "2.13.0+cu129"
+    },
+    "transformers": {
+      "record_sha256": "02b3d38575d5dc45992f859592fe143a86b04046a7c8da1b6ebc82390a90ed26",
+      "version": "5.17.0"
+    },
+    "safetensors": {
+      "record_sha256": "c5110881c0c39a4f163a23b2ccc88cad19ce1823b6dce1ceef353d33f16c004b",
+      "version": "0.8.0"
+    },
+    "vllm": {
+      "record_sha256": "bc16d6292493309ae02ca009dd57d2862e63880357a64a40be9dc6f0d6513895",
+      "version": "0.31.0+cu129"
+    },
+    "torchcodec": {
+      "record_sha256": "506253df6850b5d0af73d56c64c1e2d34c6f92cd3794a1ad1330050674855285",
+      "version": "0.17.0+cpu"
+    }
+  },
+  "source_execution_qualified": false
+}
--- /dev/null
+++ baseline/validate_reference.py
@@ -0,0 +1,135 @@
+"""Validate archive closure and lossless layout transforms; no model execution."""
+
+import collections
+import hashlib
+import json
+import pathlib
+
+import numpy as np
+import torch
+from safetensors.torch import load_file
+
+ROOT = pathlib.Path("/output/captured-reference")
+manifest = json.loads((ROOT / "manifest.json").read_text())
+runtime_source = manifest["runtime_source"]
+pins = json.loads(pathlib.Path("/probe/source-pins.json").read_text())
+assert manifest["source_pins_sha256"] == hashlib.sha256(pathlib.Path("/probe/source-pins.json").read_bytes()).hexdigest()
+for pin in runtime_source["files"].values():
+    for path in [pathlib.Path(pin["runtime_path"])]:
+        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"], path
+assert manifest["official_modeling_source"]["sha256"] == runtime_source["files"]["transformers/models/qwen3_5/modeling_qwen3_5.py"]["sha256"]
+assert manifest["source_reverified_after_capture"]
+assert manifest["limits"]["container_memory_max"] == str(12 * 1024**3)
+assert manifest["limits"]["container_cpu_max"] == "400000 100000"
+assert not manifest["runtime"]["cuda_available"]
+assert manifest["runtime"]["torch_num_threads"] == 4
+profiles = ["official-bf16-loader", "source-36-f32-restored"]
+counts = {"profiles": 0, "step_archives": 0, "canonical_domains": 0, "full_vocab_logits": 0, "observed_intermediate_tensors": 0}
+
+
+def verified_file(row):
+    path = ROOT / row["file"]
+    assert path.resolve().is_relative_to(ROOT) and not path.is_symlink() and path.is_file()
+    assert path.stat().st_size <= 32 * 1024**2
+    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
+    return path
+
+
+def verify_f32(row, expected):
+    actual = np.fromfile(verified_file(row), dtype="<f4")
+    expected = expected.detach().cpu().float().contiguous().numpy().reshape(-1)
+    # Compare bits, including zero sign; array_equal would hide that distinction.
+    assert np.array_equal(actual.view("<u4"), expected.view("<u4"))
+    assert actual.size == row["numel"] and np.isfinite(actual).all()
+
+
+for profile in profiles:
+    data = manifest["profiles"][profile]
+    audit_path = ROOT / profile / "parameter-audit.json"
+    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == data["parameter_audit_sha256"]
+    audit = json.loads(audit_path.read_text())
+    assert len(audit) == 632
+    assert all(row["exact_after_declared_dtype_cast"] for row in audit if row["loaded"] is not None)
+    excluded = [row for row in audit if row["loaded"] is None]
+    assert len(excluded) == 15 and all(row["name"].startswith("mtp.") for row in excluded)
+    original_f32 = [row for row in audit if row["source"]["dtype"] == "torch.float32"]
+    assert len(original_f32) == 36
+    assert [row["name"] for row in original_f32] == pins["raw_f32_names"]
+    assert dict(collections.Counter(row["loaded"]["dtype"] for row in audit if row["loaded"] is not None)) == data["actual_checkpoint_parameter_dtype_counts"]
+    assert len(data["steps"]) == 2
+    if profile == "source-36-f32-restored":
+        assert all(row["loaded"]["dtype"] == "torch.float32" for row in original_f32)
+    counts["profiles"] += 1
+    for step, saved in enumerate(data["steps"]):
+        raw = load_file(str(verified_file(saved["raw_archive"])), device="cpu")
+        assert len(raw) == 49
+        assert raw["logits"].shape == (1, 1, 248320)
+        assert raw["logits"].dtype == torch.bfloat16
+        assert saved["consumed_tokens"] == [11, 353][:step+1] and saved["step"] == step
+        assert saved["semantic_domain_count"] == 42 and len(saved["raw_tensor_layouts"]) == 49
+        for layer in range(24):
+            prefix = f"layer-{layer:02d}"
+            if layer % 4 == 3:
+                for key in ["keys", "values"]:
+                    assert raw[prefix + "." + key].shape == (1, 2, step+1, 256)
+                    assert raw[prefix + "." + key].dtype == torch.bfloat16
+            else:
+                assert raw[prefix + ".conv"].shape == (1, 6144, 4)
+                assert raw[prefix + ".conv"].dtype == torch.bfloat16
+                assert raw[prefix + ".recurrent"].shape == (1, 16, 128, 128)
+                assert raw[prefix + ".recurrent"].dtype == torch.float32
+        assert len(saved["semantic_domains"]) == 42
+        observed = load_file(str(verified_file(saved["observed_intermediates"])), device="cpu")
+        assert len(observed) == 15
+        assert torch.equal(observed["final_logits"].contiguous().view(torch.uint8), raw["logits"].contiguous().view(torch.uint8))
+        assert torch.equal(observed["layer-00.gdn_recurrent_fp32"].contiguous().view(torch.uint8), raw["layer-00.recurrent"].permute(0, 1, 3, 2).contiguous().view(torch.uint8))
+        assert all(bool(torch.isfinite(value).all()) for value in observed.values())
+        counts["observed_intermediate_tensors"] += len(observed)
+        verify_f32(saved["logits_f32"], raw["logits"])
+        counts["full_vocab_logits"] += 1
+        expected_domains = {(layer, kind) for layer in range(24) for kind in
+                            (["attention-kv-pair"] if layer % 4 == 3 else
+                             ["gdn-convolution-history", "gdn-recurrent"])}
+        seen = set()
+        for domain in saved["semantic_domains"]:
+            key = (domain["layer"], domain["kind"])
+            assert key in expected_domains and key not in seen
+            seen.add(key)
+            prefix = f"layer-{domain['layer']:02d}"
+            if domain["kind"] == "gdn-convolution-history":
+                expected = raw[prefix + ".conv"][..., -3:]
+            elif domain["kind"] == "gdn-recurrent":
+                expected = raw[prefix + ".recurrent"].permute(0, 1, 3, 2)
+            else:
+                key = raw[prefix + ".keys"].permute(0, 2, 1, 3).contiguous().reshape(-1)
+                val = raw[prefix + ".values"].permute(0, 2, 1, 3).contiguous().reshape(-1)
+                expected = torch.cat([key, val])
+            verify_f32(domain, expected)
+            counts["canonical_domains"] += 1
+        assert seen == expected_domains
+        assert all(bool(torch.isfinite(value).all()) for value in raw.values())
+        counts["step_archives"] += 1
+
+differences = []
+for step in range(2):
+    left_data = manifest["profiles"][profiles[0]]["steps"][step]
+    right_data = manifest["profiles"][profiles[1]]["steps"][step]
+    left = load_file(str(ROOT / left_data["raw_archive"]["file"]))
+    right = load_file(str(ROOT / right_data["raw_archive"]["file"]))
+    rows = []
+    for name in sorted(left):
+        a, b = left[name].float().contiguous().numpy(), right[name].float().contiguous().numpy()
+        changed = int(np.count_nonzero(a.view("<u4") != b.view("<u4")))
+        rows.append({"tensor": name, "elements": int(a.size), "bit_different_elements": changed, "max_abs_delta": float(np.abs(a - b).max())})
+    differences.append({"step": step, "tensors": rows})
+result = {"archive_validation": "PASS", "counts": counts, "runtime_source_files_verified": len(runtime_source["files"]), "model_or_math_parity_qualified": False, "source_revision": pins["revision"], "cross_profile_differences": differences, "scope": "archive closure, full finite tensors and exact layout derivations only; numerical profile differences are reported without a tolerance or cross-engine acceptance claim"}
+total_artifacts = 0
+for path in ROOT.rglob("*"):
+    assert not path.is_symlink()
+    if path.is_file():
+        total_artifacts += path.stat().st_size
+assert total_artifacts <= 256 * 1024**2
+assert 0 < int(manifest["cgroup_memory_peak_bytes"]) <= 12 * 1024**3
+assert counts == {"profiles": 2, "step_archives": 4, "canonical_domains": 168, "full_vocab_logits": 4, "observed_intermediate_tensors": 60}
+(ROOT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
+print(json.dumps({"archive_validation": "PASS", "counts": counts}, sort_keys=True))
--- /dev/null
+++ capture_and_validate.py
@@ -0,0 +1,7 @@
+"""ROOT ONLY: exact accepted capture integration then unchanged archive checks."""
+import gc
+import runpy
+
+runpy.run_path("/probe/capture_reference.py", run_name="__main__")
+gc.collect()
+runpy.run_path("/probe/validate_reference.py", run_name="__main__")
--- /dev/null
+++ capture_reference.py
@@ -0,0 +1,420 @@
+"""Pinned, offline CPU reference; operators remain the installed official source."""
+
+import collections
+import gc
+import signal
+import stat
+import hashlib
+import importlib.metadata
+import inspect
+import json
+import os
+import pathlib
+import platform
+import resource
+import sys
+import time
+
+# Pin the four allowed CPUs before importing the original Torch runtime.
+allowed = [int(value) for value in os.environ["ZG_PROBE_ALLOWED_CPUS"].split(",")]
+assert len(allowed) == len(set(allowed)) == 4 and set(allowed) <= os.sched_getaffinity(0)
+os.sched_setaffinity(0, allowed)
+MODE = os.environ["ZG_ATTENTION_MODE"]
+assert MODE in ("plain", "observed")
+
+import torch
+from safetensors import safe_open
+from safetensors.torch import save_file
+from transformers import AutoTokenizer
+from transformers.models.qwen3_5 import modeling_qwen3_5 as official
+
+SOURCE = pathlib.Path("/source")
+PINS = json.loads(pathlib.Path("/probe/source-pins.json").read_text())
+assert PINS["repo"] == "Qwen/Qwen3.5-2B" and PINS["revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
+AUDIT_ELEMENTS = 262144  # ≤1MiB per F32 audit operand; model operators untouched.
+ARTIFACT_CAP = 256 * 1024 * 1024
+signal.alarm(290)  # Outer owner enforces300s and always cleans its bound CID.
+resource.setrlimit(resource.RLIMIT_FSIZE, (ARTIFACT_CAP, ARTIFACT_CAP))
+assert pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip() == str(12 * 1024**3)
+assert pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip() == "400000 100000"
+assert os.environ["ATEN_CPU_CAPABILITY"] == "avx2"
+OUTPUT = pathlib.Path("/output/captured-reference")
+TOKENS = [11, 353]
+PROFILES = ["official-bf16-loader", "source-36-f32-restored"]
+torch.set_num_threads(4)
+torch.set_num_interop_threads(1)
+assert not torch.cuda.is_available()
+assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
+assert not OUTPUT.exists(), "refuse to overwrite an earlier reference"
+OUTPUT.mkdir()
+
+
+def sha(path):
+    digest = hashlib.sha256()
+    with open(path, "rb") as stream:
+        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
+            digest.update(chunk)
+    return digest.hexdigest()
+
+
+def write_json(path, data):
+    def official_json_types(value):
+        if isinstance(value, set):
+            return sorted(value)
+        raise TypeError(f"unhandled JSON type {type(value).__name__}")
+
+    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False, default=official_json_types) + "\n")
+
+
+def verify_source():
+    for item in PINS["source_files"]:
+        path = SOURCE / item["path"]
+        assert not path.is_symlink() and path.is_file() and path.stat().st_size == item["bytes"]
+        assert sha(path) == item["sha256"], item["path"]
+    return {"repo": PINS["repo"], "revision": PINS["revision"], "snapshot": PINS["snapshot"],
+            "files": PINS["source_files"], "independent_verification": PINS["root_source_validation"]}
+
+
+def audit_chunks(value):
+    assert value.device.type == "cpu" and value.is_contiguous()
+    flat = value.detach().view(-1)
+    for start in range(0, flat.numel(), AUDIT_ELEMENTS):
+        yield flat[start:start + AUDIT_ELEMENTS]
+
+
+def finite(value):
+    return all(bool(torch.isfinite(part).all()) for part in audit_chunks(value.detach().contiguous()))
+
+
+def artifact_bound():
+    total = 0
+    for path in OUTPUT.rglob("*"):
+        assert not path.is_symlink()
+        if path.is_file():
+            total += path.stat().st_size
+    assert total <= ARTIFACT_CAP, ("aggregate capture cap", total)
+    return total
+
+
+def verify_runtime():
+    expected = PINS["runtime_source"]
+    for part in expected["files"].values():
+        path = pathlib.Path(part["runtime_path"])
+        assert sha(path) == part["sha256"], str(path)
+    for name, pin in PINS["package_records"].items():
+        distribution = importlib.metadata.distribution(name)
+        assert distribution.version == pin["version"], name
+        record = distribution.read_text("RECORD")
+        assert record is not None and hashlib.sha256(record.encode()).hexdigest() == pin["record_sha256"], name
+    assert torch.__version__ == "2.13.0+cu129" and importlib.metadata.version("transformers") == "5.17.0"
+    assert torch.backends.cpu.get_cpu_capability() == "AVX2"
+    return expected
+
+
+def tensor_info(value):
+    return {
+        "dtype": str(value.dtype), "shape": list(value.shape),
+        "stride": list(value.stride()), "numel": value.numel(),
+        "bytes": value.numel() * value.element_size(),
+        "device": str(value.device), "contiguous": value.is_contiguous(),
+        "finite": finite(value),
+    }
+
+
+def f32_file(directory, name, value):
+    path = directory / (name + ".f32le.bin")
+    wide = value.detach().cpu().float().contiguous()
+    assert finite(wide)
+    # numpy explicitly requests little-endian; widening BF16 is exact.
+    path.write_bytes(wide.numpy().astype("<f4", copy=False).tobytes())
+    return {"file": str(path.relative_to(OUTPUT)), "sha256": sha(path), **tensor_info(wide)}
+
+
+def parameter_audit(model, source_file):
+    rows = []
+    with safe_open(source_file, framework="pt", device="cpu") as handle:
+        for name in sorted(handle.keys()):
+            original = handle.get_tensor(name)
+            try:
+                actual = model.get_parameter(name)
+            except AttributeError:
+                assert name.startswith("mtp."), name
+                rows.append({"name": name, "source": tensor_info(original), "loaded": None, "exact_after_declared_dtype_cast": None, "exclusion": "official Qwen3_5PreTrainedModel._keys_to_ignore_on_load_unexpected=['^mtp.*']; auxiliary MTP is absent from the ordinary conditional-generation class"})
+                continue
+            assert actual.device.type == "cpu"
+            assert actual.shape == original.shape
+            assert original.is_contiguous() and actual.is_contiguous()
+            cast_changed, cast_max = 0, 0.0
+            for left, right in zip(audit_chunks(original), audit_chunks(actual)):
+                cast = left.to(right.dtype)
+                assert torch.equal(cast.view(torch.uint8), right.view(torch.uint8)), name
+                if original.dtype != actual.dtype:
+                    delta = left.float() - right.float()
+                    cast_changed += int((delta != 0).sum())
+                    cast_max = max(cast_max, float(delta.abs().max()))
+            row = {"name": name, "source": tensor_info(original), "loaded": tensor_info(actual)}
+            row["exact_after_declared_dtype_cast"] = True
+            if original.dtype != actual.dtype:
+                row["cast_changed_elements"] = cast_changed
+                row["cast_max_abs_delta"] = cast_max
+            rows.append(row)
+    return rows
+
+
+def snapshot(cache, logits, directory, profile, step, token):
+    directory.mkdir()
+    raw = {"logits": logits.detach().cpu().contiguous().clone()}
+    raw_meta = {"logits": tensor_info(logits)}
+    domains = []
+    assert len(cache.layers) == 24
+    assert logits.shape == (1, 1, 248320)
+    for index, layer in enumerate(cache.layers):
+        prefix = f"layer-{index:02d}"
+        if hasattr(layer, "conv_states"):
+            assert set(layer.conv_states) == {0} and set(layer.recurrent_states) == {0}
+            conv, recurrent = layer.conv_states[0], layer.recurrent_states[0]
+            assert conv.shape == (1, 6144, 4)
+            assert recurrent.shape == (1, 16, 128, 128)
+            assert recurrent.dtype == torch.float32 and conv.dtype == torch.bfloat16
+            for kind, value in [("conv", conv), ("recurrent", recurrent)]:
+                raw[prefix + "." + kind] = value.detach().cpu().contiguous().clone()
+                raw_meta[prefix + "." + kind] = tensor_info(value)
+            # Official cache retains four raw projection positions. The next
+            # four-tap convolution needs the newest three as prior history.
+            domains.append({"layer": index, "kind": "gdn-convolution-history", "layout": "batch,channel,oldest-to-newest-3", "derivation": "official conv_states[0][..., -3:]", **f32_file(directory, prefix + ".conv-history", conv[..., -3:])})
+            # Source state is key-major; the Rust state reader is value-major.
+            domains.append({"layer": index, "kind": "gdn-recurrent", "layout": "batch,head,value,key", "derivation": "official recurrent_states[0].permute(0,1,3,2)", **f32_file(directory, prefix + ".recurrent-value-major", recurrent.permute(0, 1, 3, 2))})
+        else:
+            keys, values = layer.keys, layer.values
+            assert keys.shape == values.shape == (1, 2, step + 1, 256)
+            assert keys.dtype == values.dtype == torch.bfloat16
+            for kind, value in [("keys", keys), ("values", values)]:
+                raw[prefix + "." + kind] = value.detach().cpu().contiguous().clone()
+                raw_meta[prefix + "." + kind] = tensor_info(value)
+            keys = keys.permute(0, 2, 1, 3).contiguous()
+            values = values.permute(0, 2, 1, 3).contiguous()
+            pair = torch.cat([keys.reshape(-1), values.reshape(-1)])
+            domains.append({"layer": index, "kind": "attention-kv-pair", "layout": "all keys(batch,position,kv-head,width), then all values(same)", "valid_positions": step + 1, "derivation": "official keys/values permute(0,2,1,3), flatten and concatenate", **f32_file(directory, prefix + ".kv-pair", pair)})
+    assert len(domains) == 42
+    assert len(raw) == 49  # logits + 36 GDN tensors + 12 K/V tensors
+    assert all(info["finite"] for info in raw_meta.values())
+    raw_path = directory / "raw-official.safetensors"
+    save_file(raw, str(raw_path), metadata={"profile": profile, "step": str(step), "input_token": str(token), "source_revision": PINS["revision"]})
+    wide_logits = f32_file(directory, "logits", logits)
+    flat = logits.float().reshape(-1)
+    top_values, top_ids = torch.topk(flat, 10, largest=True, sorted=True)
+    result = {"profile": profile, "step": step, "input_token": token, "consumed_tokens": TOKENS[:step + 1], "cache_class": type(cache).__name__, "cache_seq_length": cache.get_seq_length(), "raw_archive": {"file": str(raw_path.relative_to(OUTPUT)), "sha256": sha(raw_path), "bytes": raw_path.stat().st_size}, "raw_tensor_layouts": raw_meta, "logits_f32": wide_logits, "argmax_token_id": int(flat.argmax()), "top10": [{"token_id": int(i), "logit": float(v)} for i, v in zip(top_ids, top_values)], "semantic_domains": domains, "semantic_domain_count": len(domains), "physical_ready_or_StateRoot_claim": False}
+    write_json(directory / "manifest.json", result)
+    return result
+
+
+started = time.time()
+runtime_source = verify_runtime()
+source_receipt = verify_source()
+write_json(OUTPUT / "source-receipt.json", source_receipt)
+source_file = SOURCE / "model.safetensors-00001-of-00001.safetensors"
+with safe_open(source_file, framework="pt", device="cpu") as source_handle:
+    names = sorted(source_handle.keys())
+    f32_names = [name for name in names if source_handle.get_slice(name).get_dtype() == "F32"]
+assert len(names) == 632 and len(f32_names) == 36
+assert f32_names == PINS["raw_f32_names"]
+assert all(".linear_attn.A_log" in name or ".linear_attn.norm.weight" in name for name in f32_names)
+assert "lm_head.weight" not in names
+print("SOURCE_VERIFIED tensors=632 original_f32=36 no_lm_head=1", flush=True)
+
+model, loading_info = official.Qwen3_5ForConditionalGeneration.from_pretrained(
+    str(SOURCE), dtype=torch.bfloat16, local_files_only=True,
+    attn_implementation="eager", use_kernels=False, output_loading_info=True,
+)
+model.eval()
+assert not loading_info.get("missing_keys") and not loading_info.get("unexpected_keys") and not loading_info.get("mismatched_keys") and not loading_info.get("error_msgs"), loading_info
+assert all(parameter.device.type == "cpu" for parameter in model.parameters())
+embedding = model.model.language_model.embed_tokens.weight
+assert model.lm_head.weight is embedding
+assert embedding.shape == (248320, 2048)
+assert model.config.text_config.hidden_size == 2048 and model.config.text_config.num_hidden_layers == 24
+tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True)
+print("OFFICIAL_MODEL_LOADED tied_head=1 cuda=0", flush=True)
+
+# Observation hooks copy actual outputs, return None and never replace an
+# operator/input/output. The down-projection input exposes the actual fused
+# SiLU(gate)*up tensor without reconstructing its arithmetic.
+intermediates = {}
+handles = []
+
+
+def remember(name, value):
+    assert name not in intermediates, name
+    assert isinstance(value, torch.Tensor)
+    intermediates[name] = value.detach().cpu().contiguous().clone()
+
+
+def output_hook(module, name):
+    handles.append(module.register_forward_hook(lambda _module, _args, value: remember(name, value)))
+
+
+def input_hook(module, name):
+    handles.append(module.register_forward_pre_hook(lambda _module, args: remember(name, args[0])))
+
+
+from attention_observation_hook import AttentionProjectionObserver
+attention = None
+try:
+    language = model.model.language_model
+    layer0 = language.layers[0]
+    output_hook(language.embed_tokens, "embedding")
+    output_hook(layer0.input_layernorm, "layer-00.attention_input_norm")
+    output_hook(layer0.linear_attn.in_proj_qkv, "layer-00.gdn_qkv")
+    output_hook(layer0.linear_attn, "layer-00.attention_output")
+    input_hook(layer0.post_attention_layernorm, "layer-00.post_attention_residual")
+    output_hook(layer0.post_attention_layernorm, "layer-00.ffn_input_norm")
+    output_hook(layer0.mlp.gate_proj, "layer-00.ffn_gate")
+    output_hook(layer0.mlp.up_proj, "layer-00.ffn_up")
+    input_hook(layer0.mlp.down_proj, "layer-00.ffn_fused")
+    output_hook(layer0.mlp.down_proj, "layer-00.ffn_output")
+    output_hook(layer0, "layer-00.post_ffn_residual")
+    output_hook(language.norm, "final_norm")
+
+    manifest = {
+        "schema": "zerogravity.official-cpu-qwen35-reference.v2-hf2b",
+        "source_pins_sha256": sha(pathlib.Path("/probe/source-pins.json")),
+        "source_repo": source_receipt["repo"], "source_revision": source_receipt["revision"],
+        "image": PINS["derived_image_id"], "base_image": PINS["base_image"],
+        "runtime_source": runtime_source, "package_records": PINS["package_records"],
+        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "safetensors": importlib.metadata.version("safetensors"), "vllm": importlib.metadata.version("vllm"), "torch_num_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "cuda_available": torch.cuda.is_available(), "mkldnn_enabled": torch.backends.mkldnn.enabled, "default_dtype": str(torch.get_default_dtype()), "cpu_capability": torch.backends.cpu.get_cpu_capability(), "machine": platform.machine(), "processor": platform.processor()},
+        "loader": {"model_class": type(model).__name__, "dtype": "torch.bfloat16", "attn_implementation": "eager", "use_kernels": False, "local_files_only": True, "loading_info": loading_info},
+        "config": model.config.to_dict(), "tokenizer": {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer), "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id, "tokens": TOKENS, "token_strings": tokenizer.convert_ids_to_tokens(TOKENS), "decoded": tokenizer.decode(TOKENS, skip_special_tokens=False)},
+        "head": {"tied_parameter_object": True, "source_has_lm_head_tensor": False, "embedding_name": "model.language_model.embed_tokens.weight", "output_dtype_rule": "official lm_head(hidden); no loss, no automatic FP32 upcast"},
+        "protocol": "one cached official forward per fixed input token, no sampling or generation; first token uses chunk reference, next token recurrent reference",
+        "limits": {"container_memory_max": pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip(), "container_cpu_max": pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip(), "network": "none", "source_mount": "read-only", "gpu_exposed": False},
+        "scope": "Official Transformers eager CPU attention and slow PyTorch GDN operators only. No GPU FLA, vLLM inference, throughput, quantization, Rust bit parity, or trained27B qualification.",
+        "profiles": {},
+        "source_geometry": {"hidden": 2048, "ffn": 6144, "layers": 24, "gdn_layers": 18, "full_attention_layers": 6, "gdn_heads": 16, "gdn_head_width": 128, "kv_heads": 2, "kv_head_width": 256},
+    }
+    model_source = pathlib.Path(inspect.getsourcefile(official))
+    manifest["official_modeling_source"] = {"path": str(model_source), "sha256": sha(model_source)}
+    manifest["capture_script_sha256"] = sha(pathlib.Path(__file__))
+
+    if MODE == "observed":
+        attention = AttentionProjectionObserver(model, official)
+    for profile in PROFILES:
+        if profile == "source-36-f32-restored":
+            with safe_open(source_file, framework="pt", device="cpu") as source_handle:
+                for name in f32_names:
+                    parameter = model.get_parameter(name)
+                    parameter.data = source_handle.get_tensor(name).clone()
+                    assert parameter.dtype == torch.float32
+            print("RESTORED_ORIGINAL_F32_PARAMETERS count=36", flush=True)
+        profile_dir = OUTPUT / profile
+        profile_dir.mkdir()
+        audit = parameter_audit(model, source_file)
+        write_json(profile_dir / "parameter-audit.json", audit)
+        assert len(audit) == 632 and sum(row["loaded"] is None for row in audit) == 15
+        dtype_counts = collections.Counter(row["loaded"]["dtype"] for row in audit if row["loaded"] is not None)
+        print("PROFILE_AUDIT", profile, dict(dtype_counts), flush=True)
+        profile_result = {"profile": profile, "parameter_audit_sha256": sha(profile_dir / "parameter-audit.json"), "source_parameter_count": len(audit), "actual_checkpoint_parameter_dtype_counts": dict(dtype_counts), "source_f32_parameter_names": f32_names, "restoration": "none; unmodified official requested-BF16 loader" if profile == PROFILES[0] else "only the 36 originally F32 checkpoint parameters restored to their exact source dtype and bytes; official operators unchanged", "steps": []}
+        cache = None
+        with torch.inference_mode():
+            for step, token in enumerate(TOKENS):
+                intermediates = {}
+                calls = collections.Counter()
+                signatures = {}
+
+                def observe_calls(frame, event, arg):
+                    if attention is not None:
+                        attention.observe_source_return(frame, event, arg)
+                    if frame.f_code.co_filename != str(model_source):
+                        return
+                    name = frame.f_code.co_name
+                    if event == "return" and name in {"causal_conv1d_fn", "causal_conv1d_update"}:
+                        caller = frame.f_back
+                        while caller is not None:
+                            owner = caller.f_locals.get("self")
+                            if isinstance(owner, official.Qwen3_5GatedDeltaNet):
+                                if owner.layer_idx == 0:
+                                    remember("layer-00.gdn_conv_silu", arg.transpose(1, 2))
+                                break
+                            caller = caller.f_back
+                        return
+                    if event != "call":
+                        return
+                    if name in {"torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update", "eager_attention_forward"}:
+                        key = name + ":" + str(frame.f_code.co_firstlineno)
+                        calls[key] += 1
+                        signatures[key] = {key: tensor_info(value) for key, value in frame.f_locals.items() if isinstance(value, torch.Tensor)}
+
+                if attention is not None:
+                    attention.begin_step(profile, step, token)
+                step_start = time.time()
+                sys.setprofile(observe_calls)
+                try:
+                    result = model(input_ids=torch.tensor([[token]], dtype=torch.int64), past_key_values=cache, use_cache=True, return_dict=True)
+                finally:
+                    sys.setprofile(None)
+                cache = result.past_key_values
+                expected = {("torch_chunk_gated_delta_rule" if step == 0 else "torch_recurrent_gated_delta_rule"): 18, ("causal_conv1d_fn" if step == 0 else "causal_conv1d_update"): 18, "eager_attention_forward": 6}
+                observed_by_name = {key.split(":")[0]: value for key, value in calls.items()}
+                assert observed_by_name == expected, (expected, observed_by_name)
+                step_result = snapshot(cache, result.logits, profile_dir / f"step-{step}", profile, step, token)
+                if attention is not None:
+                    attention.finish_step(profile_dir / f"step-{step}" / "attention-observation")
+                remember("final_logits", result.logits)
+                remember("layer-00.gdn_recurrent_fp32", cache.layers[0].recurrent_states[0].permute(0, 1, 3, 2))
+                assert len(intermediates) == 15
+                intermediate_path = profile_dir / f"step-{step}" / "observed-intermediates.safetensors"
+                save_file(intermediates, str(intermediate_path), metadata={"profile": profile, "step": str(step), "scope": "actual layer0/module input/output copies; no reconstructed arithmetic"})
+                step_result["observed_intermediates"] = {"file": str(intermediate_path.relative_to(OUTPUT)), "sha256": sha(intermediate_path), "bytes": intermediate_path.stat().st_size, "tensors": {name: tensor_info(value) for name, value in intermediates.items()}, "hook_scope": "official module pre/post hooks, CPU tensor clone only; slow causal-conv source return observed with sys.setprofile; no replacement return"}
+                step_result["observed_official_operator_calls"] = dict(calls)
+                step_result["last_observed_operator_tensor_arguments"] = signatures
+                step_result["diagnostic_elapsed_seconds_not_benchmark"] = time.time() - step_start
+                write_json(profile_dir / f"step-{step}" / "manifest.json", step_result)
+                profile_result["steps"].append(step_result)
+                artifact_bound()
+                print("REFERENCE_STEP_COMPLETE", profile, "step", step, "argmax", step_result["argmax_token_id"], "domains", len(step_result["semantic_domains"]), "seconds", step_result["diagnostic_elapsed_seconds_not_benchmark"], flush=True)
+        manifest["profiles"][profile] = profile_result
+        write_json(OUTPUT / "manifest.json", manifest)
+        del result, cache
+
+finally:
+    # Every installed hook is removed even after forward/capture/receipt failure.
+    primary = sys.exception()
+    removal = []
+    try:
+        sys.setprofile(None)
+    except BaseException as error:
+        removal.append(error)
+    if attention is not None:
+        try:
+            attention.close()
+        except BaseException as error:
+            removal.append(error)
+    attention = None  # Release observed module owners before archive validation.
+    for handle in reversed(handles):
+        try:
+            handle.remove()
+        except BaseException as error:
+            removal.append(error)
+    handles.clear()
+    if removal:
+        raise BaseExceptionGroup("capture and all hook removal failures",
+                                 ([primary] if primary is not None else []) + removal) from None
+del model, embedding, language, layer0, intermediates
+gc.collect()
+assert verify_source() == source_receipt
+assert verify_runtime() == runtime_source
+manifest["artifact_bytes_before_final_manifest"] = artifact_bound()
+manifest["cgroup_memory_peak_bytes"] = pathlib.Path("/sys/fs/cgroup/memory.peak").read_text().strip()
+manifest["source_reverified_after_capture"] = True
+manifest["elapsed_seconds_not_benchmark"] = time.time() - started
+manifest["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
+write_json(pathlib.Path("/output/attention-mode.json"), {
+    "mode": MODE, "owned_hooks_removed": True, "profiler_retired": sys.getprofile() is None,
+    "cpu_affinity": sorted(os.sched_getaffinity(0)), "teacher_tokens": TOKENS,
+    "profiles": PROFILES, "observer_hook_sha256": sha(pathlib.Path("/probe/attention_observation_hook.py")),
+    "accepted_capture_sha256": "6ee140185766dee237ea4df802504ef7b7be80110f5566876a644414cf66049d",
+    "expected_attention_tensor_count": 216 if MODE == "observed" else 0,
+    "full_model_or_Rust_or_GPU_parity_qualified": False,
+})
+write_json(OUTPUT / "manifest.json", manifest)
+artifact_bound()
+print("REFERENCE_CAPTURE_COMPLETE", json.dumps({"profiles": list(manifest["profiles"]), "steps_each": 2, "full_vocab": 248320, "semantic_domains_each": 42, "max_rss_kib": manifest["max_rss_kib"]}), flush=True)
--- /dev/null
+++ owned_container_v4.py
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
+++ parent-pins.json
@@ -0,0 +1,49 @@
+{
+  "accepted_capture": {
+    "bytes": 22266,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/capture_reference.py",
+    "sha256": "6ee140185766dee237ea4df802504ef7b7be80110f5566876a644414cf66049d"
+  },
+  "accepted_source_pins": {
+    "bytes": 10079,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/source-pins.json",
+    "sha256": "9bd17bb4d58582373034fd7b0097e36d3e39bf91d22bce837cb7fe109ee9fee4"
+  },
+  "accepted_validator": {
+    "bytes": 7906,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/validate_reference.py",
+    "sha256": "e29ea2a21c850cb4e3210f4206412282fb857c40344c6338c7604bf967e8fd72"
+  },
+  "attention_hook": {
+    "bytes": 9970,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-attention-slice-20261020/observation-prep/attention_observation_hook.py",
+    "sha256": "12407465ee6795c2c94d191f1ff3d5550b7bef29e8d916f8f824b6b736dd9495"
+  },
+  "codec_manifest": "sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
+  "image_id": "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
+  "legacy_metadata": [
+    {
+      "bytes": 3829,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/root-probe-2/invocation.json",
+      "sha256": "c601fa7e495807332e57c80a177276f57f35c90430d8c82a5691f453781c8b72"
+    },
+    {
+      "bytes": 313368,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/root-probe-2/captured-reference/manifest.json",
+      "sha256": "650a09dd392206231e21c2ea92064fc9e71497b1334c2db362fd3e097a4d1e2d"
+    },
+    {
+      "bytes": 17355,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/root-probe-2/captured-reference/validation.json",
+      "sha256": "6d7d6ba1a16555c4930c922c3bcd6a6c8d957e6c231b3d9b5b6924dd54a3ee92"
+    }
+  ],
+  "legacy_reference_root": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/root-probe-2/captured-reference",
+  "owned_lifetime_helper": {
+    "bytes": 11840,
+    "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-vllm-gemm-observation-20261020/owned_container_v4.py",
+    "sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
+  },
+  "raw_payloads_hashed_or_loaded_by_author": false,
+  "scope": "PREPARED_ONLY_NO_RUNTIME_BY_AUTHOR"
+}
--- /dev/null
+++ run_pair.py
@@ -0,0 +1,139 @@
+"""ROOT ONLY: serialized plain/observed original CPU capture; no GPU exposure."""
+import argparse
+import importlib.util
+import json
+import os
+from pathlib import Path
+import subprocess
+import time
+import uuid
+
+from support import HERE, read_json, require, sha, write
+
+IMAGE = "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe"
+CODEC_MANIFEST = "sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634"
+HELPER_SHA = "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
+ENTRY_FILES = ("support.py", "source-pins.json", "parent-pins.json", "capture_reference.py",
+               "validate_reference.py", "capture_and_validate.py", "attention_observation_hook.py",
+               "owned_container_v4.py", "run_pair.py", "validate_pair.py")
+
+
+def reviewed_helper():
+    path = HERE / "owned_container_v4.py"
+    require(sha(path) == HELPER_SHA, "exact reviewed GEMM owned_container_v4 lifetime")
+    spec = importlib.util.spec_from_file_location("owned_cpu_attention", path)
+    helper = importlib.util.module_from_spec(spec)
+    spec.loader.exec_module(helper)
+    return helper
+
+
+def run_mode(helper, output, source, mode, cpus):
+    directory = output / mode
+    directory.mkdir(mode=0o700)
+    nonce = uuid.uuid4().hex
+    name = "zg-hf2b-attention-" + mode + "-" + nonce
+    cidfile = directory / "owned-container.cid"
+    environment = {
+        "ZG_PROBE_ALLOWED_CPUS": ",".join(map(str, cpus)), "ZG_ATTENTION_MODE": mode,
+        "PYTHONPATH": "/probe", "PYTHONDONTWRITEBYTECODE": "1",
+        "CUDA_VISIBLE_DEVICES": "-1", "NVIDIA_VISIBLE_DEVICES": "void",
+        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_PLUGINS": "",
+        "TOKENIZERS_PARALLELISM": "false", "ATEN_CPU_CAPABILITY": "avx2",
+        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
+        "LD_PRELOAD": "", "LD_AUDIT": "",
+    }
+    argv = helper.PODMAN + ["run", "--name", name, "--label", helper.LABEL + "=" + nonce,
+             "--cidfile", str(cidfile), "--rm", "--pull=never", "--network=none", "--read-only",
+             "--memory=12g", "--memory-swap=12g", "--cpus=4", "--pids-limit=256",
+             "--cap-drop=all", "--security-opt=no-new-privileges",
+             "--ulimit", "fsize=268435456:268435456", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g"]
+    for key, value in environment.items():
+        argv += ["-e", key + "=" + value]
+    for mount, target in ((HERE, "/probe:ro"), (source, "/source:ro"), (directory, "/output:rw")):
+        argv += ["-v", str(mount) + ":" + target]
+    argv += ["--entrypoint", "python3", IMAGE, "/probe/capture_and_validate.py"]
+    receipt = {"status": "UNQUALIFIED", "mode": mode, "argv": argv, "owned_name": name,
+               "owned_nonce": nonce, "image_id": IMAGE, "expected_cpus": cpus,
+               "entry_files": {name: sha(HERE / name) for name in ENTRY_FILES},
+               "owned_lifetime_helper_sha256": HELPER_SHA,
+               "scope": "original CPU paired observation only; no Rust/model/profile eligibility"}
+    write(directory / "invocation.json", receipt)
+    child = log = failure = None
+    started = time.monotonic()
+    try:
+        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
+                                 stderr=subprocess.STDOUT, start_new_session=True)
+        log = helper.Log(child.stdout, directory / "execution.log")
+        receipt["owned_cli_pid"] = child.pid
+        write(directory / "invocation.json", receipt)
+        receipt["exit_code"] = log.observe(child, 300)
+    except BaseException as caught:
+        failure = caught
+    finally:
+        receipt["cleanup_errors"] = helper.cleanup(child, name, nonce, cidfile, receipt) if child is not None else []
+        try:
+            if log is not None:
+                log.finish()
+        except BaseException as caught:
+            receipt["cleanup_errors"].append(repr(caught))
+        finally:
+            if log is not None:
+                receipt["log_cleanup"] = {"pipe_eof": log.eof, "all_closed": log.closed,
+                                          "total_bytes": log.total, "retained_bytes": log.retained}
+        if log is None and child is not None and child.stdout is not None:
+            try:
+                child.stdout.close()
+            except BaseException as caught:
+                receipt["cleanup_errors"].append("uninitialized log pipe close: " + repr(caught))
+        receipt["elapsed_seconds_not_benchmark"] = time.monotonic() - started
+    if failure is not None:
+        receipt["failure"] = repr(failure)
+    if failure is None and receipt.get("exit_code") == 0 and not receipt["cleanup_errors"]:
+        receipt["status"] = "CAPTURE_COMPLETED_REQUIRES_VALIDATION"
+    try:
+        write(directory / "invocation.json", receipt)
+    except BaseException as caught:
+        raise RuntimeError("final receipt I/O failed: " + repr(caught) + "; primary=" + repr(failure)
+                           + "; cleanup=" + repr(receipt["cleanup_errors"])) from caught
+    require(receipt["status"] == "CAPTURE_COMPLETED_REQUIRES_VALIDATION",
+            "owned CPU capture/cleanup failed; inspect " + str(directory))
+
+
+def main():
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--output", required=True, type=Path, help="fresh directory beneath this prepared run")
+    options = parser.parse_args()
+    require(options.output.is_absolute() and options.output.parent.resolve() == HERE,
+            "fresh absolute output directly beneath this owned run")
+    frozen = read_json(HERE / "freeze-manifest.json")
+    for pin in frozen["files"]:
+        path = HERE / pin["file"]
+        require(path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"], "frozen preparation")
+    parents = read_json(HERE / "parent-pins.json")
+    for pin in parents["legacy_metadata"] + [parents[key] for key in
+            ("accepted_capture", "accepted_validator", "accepted_source_pins", "attention_hook", "owned_lifetime_helper")]:
+        path = Path(pin["path"])
+        require(path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"], "original parent evidence unchanged")
+    pins = read_json(HERE / "source-pins.json")
+    require(pins["repo"] == "Qwen/Qwen3.5-2B" and pins["revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc",
+            "exact original source")
+    source = Path(pins["source_directory"])
+    require(source.is_dir(), "already acquired original source")
+    helper = reviewed_helper()
+    inspected = helper.command(helper.PODMAN + ["image", "inspect", IMAGE], 15)
+    require(inspected.returncode == 0, "existing exact image; never pull")
+    image, = json.loads(inspected.stdout)
+    require(image["Id"] == IMAGE.removeprefix("sha256:") and image["Digest"] == CODEC_MANIFEST,
+            "exact approved codec-only image")
+    cpus = sorted(os.sched_getaffinity(0))[:4]
+    require(len(cpus) == 4, "four actually allowed CPUs")
+    options.output.mkdir(mode=0o700)
+    write(options.output / "image-inspect.json", image)
+    for mode in ("plain", "observed"):
+        run_mode(helper, options.output, source, mode, cpus)
+    from validate_pair import bounded_validate
+    print(json.dumps(bounded_validate(options.output), indent=2, sort_keys=True))
+
+
+if __name__ == "__main__":
+    main()
--- /dev/null
+++ source-pins.json
@@ -0,0 +1,226 @@
+{
+  "schema": "zerogravity.hf2b-reference-pins.v1",
+  "repo": "Qwen/Qwen3.5-2B",
+  "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
+  "snapshot": "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52",
+  "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52/files",
+  "source_files": [
+    {
+      "path": "LICENSE",
+      "bytes": 11544,
+      "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "README.md",
+      "bytes": 62814,
+      "sha256": "c0e83a849c776e6fa843d011f023132d942a0f0140d903205bf1c363adad2275",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "chat_template.jinja",
+      "bytes": 7755,
+      "sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "config.json",
+      "bytes": 2908,
+      "sha256": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "merges.txt",
+      "bytes": 3353259,
+      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "model.safetensors-00001-of-00001.safetensors",
+      "bytes": 4548221488,
+      "sha256": "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
+      "provenance": "independent official LFS SHA256"
+    },
+    {
+      "path": "model.safetensors.index.json",
+      "bytes": 64460,
+      "sha256": "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "preprocessor_config.json",
+      "bytes": 390,
+      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "tokenizer.json",
+      "bytes": 12807982,
+      "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
+      "provenance": "independent official LFS SHA256"
+    },
+    {
+      "path": "tokenizer_config.json",
+      "bytes": 16709,
+      "sha256": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "video_preprocessor_config.json",
+      "bytes": 385,
+      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "vocab.json",
+      "bytes": 6722759,
+      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    }
+  ],
+  "raw_f32_names": [
+    "model.language_model.layers.0.linear_attn.A_log",
+    "model.language_model.layers.0.linear_attn.norm.weight",
+    "model.language_model.layers.1.linear_attn.A_log",
+    "model.language_model.layers.1.linear_attn.norm.weight",
+    "model.language_model.layers.10.linear_attn.A_log",
+    "model.language_model.layers.10.linear_attn.norm.weight",
+    "model.language_model.layers.12.linear_attn.A_log",
+    "model.language_model.layers.12.linear_attn.norm.weight",
+    "model.language_model.layers.13.linear_attn.A_log",
+    "model.language_model.layers.13.linear_attn.norm.weight",
+    "model.language_model.layers.14.linear_attn.A_log",
+    "model.language_model.layers.14.linear_attn.norm.weight",
+    "model.language_model.layers.16.linear_attn.A_log",
+    "model.language_model.layers.16.linear_attn.norm.weight",
+    "model.language_model.layers.17.linear_attn.A_log",
+    "model.language_model.layers.17.linear_attn.norm.weight",
+    "model.language_model.layers.18.linear_attn.A_log",
+    "model.language_model.layers.18.linear_attn.norm.weight",
+    "model.language_model.layers.2.linear_attn.A_log",
+    "model.language_model.layers.2.linear_attn.norm.weight",
+    "model.language_model.layers.20.linear_attn.A_log",
+    "model.language_model.layers.20.linear_attn.norm.weight",
+    "model.language_model.layers.21.linear_attn.A_log",
+    "model.language_model.layers.21.linear_attn.norm.weight",
+    "model.language_model.layers.22.linear_attn.A_log",
+    "model.language_model.layers.22.linear_attn.norm.weight",
+    "model.language_model.layers.4.linear_attn.A_log",
+    "model.language_model.layers.4.linear_attn.norm.weight",
+    "model.language_model.layers.5.linear_attn.A_log",
+    "model.language_model.layers.5.linear_attn.norm.weight",
+    "model.language_model.layers.6.linear_attn.A_log",
+    "model.language_model.layers.6.linear_attn.norm.weight",
+    "model.language_model.layers.8.linear_attn.A_log",
+    "model.language_model.layers.8.linear_attn.norm.weight",
+    "model.language_model.layers.9.linear_attn.A_log",
+    "model.language_model.layers.9.linear_attn.norm.weight"
+  ],
+  "tensor_counts": {
+    "F32": 36,
+    "BF16": 596
+  },
+  "dtype_elements": {
+    "F32": 2592,
+    "BF16": 2274067232
+  },
+  "root_source_validation": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-source-plan-20261020/root-post-acquisition-validation.json",
+    "bytes": 5496,
+    "sha256": "57929dd60414c1f2b4df536c05f39bcbef9957221ae45ae42f7982d6d1666ccb"
+  },
+  "cache_manifest": {
+    "path": "/home/pvl/.cache/abird/zerogravity/models/snapshots/88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52/manifest.json",
+    "bytes": 2246,
+    "sha256": "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52"
+  },
+  "base_capture": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/capture_reference.py",
+    "bytes": 18717,
+    "sha256": "225d8637f927f8bc6ce99af9d9dda6a6a5cd33fc9d7d24042dda7b98297f335d"
+  },
+  "base_validator": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/validate_reference.py",
+    "bytes": 5573,
+    "sha256": "5ef1077ed6abdb37093701cb2846fc044a35f62eb9d883891876c5e98f0ced09"
+  },
+  "runtime_source_manifest": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/runtime-source-manifest.json",
+    "bytes": 1942,
+    "sha256": "24eabe68b10c2ac6a17e2dcdcd53805abe47556a98b8b6a4f2f1c27c60e6c673"
+  },
+  "runtime_source": {
+    "python": "3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]",
+    "versions": {
+      "transformers": "5.17.0",
+      "torch": "2.13.0+cu129",
+      "safetensors": "0.8.0",
+      "vllm": "0.31.0+cu129",
+      "flash-linear-attention": null,
+      "fla-core": null,
+      "triton": "3.7.1"
+    },
+    "files": {
+      "transformers/models/qwen3_5/modeling_qwen3_5.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py",
+        "sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b",
+        "bytes": 95685,
+        "artifact": "pinned-runtime-source/modeling_qwen3_5.py"
+      },
+      "transformers/models/qwen3_5/configuration_qwen3_5.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_5/configuration_qwen3_5.py",
+        "sha256": "19966c3200cee92cc4bccaef5f94550c56d731dbf03858a9bd2ed6aebcc3f7da",
+        "bytes": 8395,
+        "artifact": "pinned-runtime-source/configuration_qwen3_5.py"
+      },
+      "transformers/cache_utils.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/cache_utils.py",
+        "sha256": "702144bb44553f6339ea1bf23c8205a708bb5f8c7c09cb3a2db484182646743c",
+        "bytes": 104677,
+        "artifact": "pinned-runtime-source/cache_utils.py"
+      },
+      "transformers/modeling_utils.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py",
+        "sha256": "38c2bd02ed7af229f54e2f02dae65663be7af29ed2c9498bd1c460cf60fbb62d",
+        "bytes": 268806,
+        "artifact": "pinned-runtime-source/modeling_utils.py"
+      },
+      "transformers/integrations/hub_kernels.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/integrations/hub_kernels.py",
+        "sha256": "c1eafefe0cbdf7f0deaa21b1ef6cb6d7c6082b3f486651d069ff38fcf3e53c95",
+        "bytes": 44974,
+        "artifact": "pinned-runtime-source/hub_kernels.py"
+      }
+    }
+  },
+  "derived_image_receipt": {
+    "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/torchcodec-cpu/derived-image-receipt.json",
+    "bytes": 7655,
+    "sha256": "c53c8421af4c832b901c00f4aabb65c7a1901b92c0655af188fa886fdaac63c0"
+  },
+  "derived_image_id": "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
+  "base_image": "docker.io/vllm/vllm-openai@sha256:b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f",
+  "package_records": {
+    "torch": {
+      "record_sha256": "4ae8e32005981a22c51a85130862a0ca173a01441a9de439b405a380c8dba243",
+      "version": "2.13.0+cu129"
+    },
+    "transformers": {
+      "record_sha256": "02b3d38575d5dc45992f859592fe143a86b04046a7c8da1b6ebc82390a90ed26",
+      "version": "5.17.0"
+    },
+    "safetensors": {
+      "record_sha256": "c5110881c0c39a4f163a23b2ccc88cad19ce1823b6dce1ceef353d33f16c004b",
+      "version": "0.8.0"
+    },
+    "vllm": {
+      "record_sha256": "bc16d6292493309ae02ca009dd57d2862e63880357a64a40be9dc6f0d6513895",
+      "version": "0.31.0+cu129"
+    },
+    "torchcodec": {
+      "record_sha256": "506253df6850b5d0af73d56c64c1e2d34c6f92cd3794a1ad1330050674855285",
+      "version": "0.17.0+cpu"
+    }
+  },
+  "source_execution_qualified": false
+}
--- /dev/null
+++ static-checks.json
@@ -0,0 +1,39 @@
+{
+  "accepted_named_function_changed_only_by_observer_delegation": [
+    "observe_calls"
+  ],
+  "accepted_named_functions_unchanged": [
+    "sha",
+    "write_json",
+    "verify_source",
+    "audit_chunks",
+    "finite",
+    "artifact_bound",
+    "verify_runtime",
+    "tensor_info",
+    "f32_file",
+    "parameter_audit",
+    "snapshot",
+    "remember",
+    "output_hook",
+    "input_hook",
+    "official_json_types"
+  ],
+  "author_executed_models_operators_containers_tests": false,
+  "original_model_call_AST_unchanged": true,
+  "owned_helper_sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d",
+  "python_AST_files": [
+    "attention_observation_hook.py",
+    "capture_and_validate.py",
+    "capture_reference.py",
+    "owned_container_v4.py",
+    "run_pair.py",
+    "support.py",
+    "test_validate_pair.py",
+    "validate_pair.py",
+    "validate_reference.py"
+  ],
+  "scope": "AST_AND_SOURCE_BYTES_ONLY",
+  "tests_prepared_not_executed": 10,
+  "unchanged_original_validator_and_source_pins": true
+}
--- /dev/null
+++ support.py
@@ -0,0 +1,34 @@
+"""Bounded preparation support; no model/container side effects on import."""
+import hashlib
+import json
+import os
+from pathlib import Path
+
+HERE = Path(__file__).resolve().parent
+
+
+def require(value, message):
+    if not value:
+        raise ValueError(message)
+
+
+def sha(path):
+    digest = hashlib.sha256()
+    with Path(path).open("rb") as stream:
+        for block in iter(lambda: stream.read(1024 * 1024), b""):
+            digest.update(block)
+    return digest.hexdigest()
+
+
+def write(path, value):
+    with Path(path).open("w", encoding="utf-8") as stream:
+        os.fchmod(stream.fileno(), 0o600)
+        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
+        stream.write("\n")
+
+
+def read_json(path, cap=1024 * 1024):
+    path = Path(path)
+    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= cap,
+            "bounded regular JSON file: " + str(path))
+    return json.loads(path.read_text())
--- /dev/null
+++ test_validate_pair.py
@@ -0,0 +1,136 @@
+"""SOURCE-FREE fixtures, PREPARED ONLY; ROOT may execute after review."""
+import hashlib
+import json
+from pathlib import Path
+import struct
+import tempfile
+import unittest
+
+from validate_pair import LAYERS, MODELING_SHA, STAGES, TORCH_SOURCE, archive, equal_archives, integer, observation
+
+
+def make_archive(path, values, *, gap=False):
+    header, payload = {}, b"\0\0" if gap else b""
+    for name, (dtype, shape, raw) in sorted(values.items()):
+        lo = len(payload)
+        payload += raw
+        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [lo, len(payload)]}
+    encoded = json.dumps(header, separators=(",", ":")).encode()
+    encoded += b" " * ((-len(encoded)) % 8)
+    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
+
+
+def fixture(directory, *, word=0):
+    values, metadata = {}, {}
+    total = 0
+    for layer in LAYERS:
+        for stage, shape in STAGES.items():
+            name = f"layer-{layer:02d}.{stage}"
+            elements, stride, strides = 1, 1, []
+            for width in reversed(shape):
+                strides.insert(0, stride)
+                stride *= width
+                elements *= width
+            raw = struct.pack("<H", word) * elements
+            values[name] = ("BF16", shape, raw)
+            original = list(strides)
+            if stage == "query":
+                original[-2] = 512
+            metadata[name] = {"dtype": "torch.bfloat16", "shape": shape,
+                              "original_stride": original, "saved_stride": strides,
+                              "bytes": len(raw), "source_module_layer": layer, "stage": stage}
+            total += len(raw)
+    path = directory / "attention.safetensors"
+    make_archive(path, values)
+    report = {"status": "OBSERVED_OPERATORS_ONLY_FULL_TRANSPARENCY_UNQUALIFIED",
+              "profile": "official-bf16-loader", "step": 0, "teacher_token": 11,
+              "modeling_sha256": MODELING_SHA, "torch_source": TORCH_SOURCE,
+              "no_model_or_state_or_provider_eligibility": True,
+              "tensor_count": 54, "tensors": metadata, "aggregate_payload_bytes": total,
+              "archive": {"file": path.name, "bytes": path.stat().st_size,
+                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
+    (directory / "observation.json").write_text(json.dumps(report))
+    return report, total
+
+
+class ComparatorFixtures(unittest.TestCase):
+    def test_identical_raw_bits(self):
+        with tempfile.TemporaryDirectory() as folder:
+            a, b = Path(folder) / "a", Path(folder) / "b"
+            for path in (a, b):
+                make_archive(path, {"tensor": ("BF16", [2], b"\0\0\0\x80")})
+            self.assertEqual(equal_archives(a, b, 1), 4)
+
+    def test_signed_zero_difference_is_refused(self):
+        with tempfile.TemporaryDirectory() as folder:
+            a, b = Path(folder) / "a", Path(folder) / "b"
+            make_archive(a, {"tensor": ("BF16", [1], b"\0\0")})
+            make_archive(b, {"tensor": ("BF16", [1], b"\0\x80")})
+            with self.assertRaisesRegex(ValueError, "exact source raw bytes"):
+                equal_archives(a, b, 1)
+
+    def test_payload_gap_is_refused(self):
+        with tempfile.TemporaryDirectory() as folder:
+            path = Path(folder) / "a"
+            make_archive(path, {"tensor": ("BF16", [1], b"\0\0")}, gap=True)
+            with self.assertRaisesRegex(ValueError, "without gaps"):
+                archive(path)
+
+    def test_namespace_omission_is_refused(self):
+        with tempfile.TemporaryDirectory() as folder:
+            a, b = Path(folder) / "a", Path(folder) / "b"
+            make_archive(a, {"one": ("BF16", [1], b"\0\0")})
+            make_archive(b, {"another": ("BF16", [1], b"\0\0")})
+            with self.assertRaisesRegex(ValueError, "namespace"):
+                equal_archives(a, b, 1)
+
+    def test_complete_synthetic_stage_contract(self):
+        with tempfile.TemporaryDirectory() as folder:
+            path = Path(folder)
+            _, total = fixture(path)
+            self.assertEqual(total, 172032)
+            self.assertEqual(observation(path, "official-bf16-loader", 0, 0), total)
+
+    def test_missing_stage_is_refused(self):
+        with tempfile.TemporaryDirectory() as folder:
+            path = Path(folder)
+            row, _ = fixture(path)
+            del row["tensors"]["layer-23.gate"]
+            (path / "observation.json").write_text(json.dumps(row))
+            with self.assertRaisesRegex(ValueError, "all six layers"):
+                observation(path, "official-bf16-loader", 0, 0)
+
+    def test_profile_and_teacher_misbinding_is_refused(self):
+        with tempfile.TemporaryDirectory() as folder:
+            path = Path(folder)
+            row, _ = fixture(path)
+            row["teacher_token"] = 353
+            (path / "observation.json").write_text(json.dumps(row))
+            with self.assertRaisesRegex(ValueError, "source/protocol"):
+                observation(path, "official-bf16-loader", 0, 0)
+            with self.assertRaises(ValueError):
+                observation(path, "source-36-f32-restored", 0, 0)
+
+    def test_nonfinite_raw_observation_is_refused(self):
+        with tempfile.TemporaryDirectory() as folder:
+            path = Path(folder)
+            fixture(path, word=0x7fc1)
+            with self.assertRaisesRegex(ValueError, "finite actual BF16"):
+                observation(path, "official-bf16-loader", 0, 0)
+
+    def test_strided_query_role_is_required(self):
+        with tempfile.TemporaryDirectory() as folder:
+            path = Path(folder)
+            row, _ = fixture(path)
+            row["tensors"]["layer-03.query"]["original_stride"][-2] = 256
+            (path / "observation.json").write_text(json.dumps(row))
+            with self.assertRaisesRegex(ValueError, "interleaved query"):
+                observation(path, "official-bf16-loader", 0, 0)
+
+    def test_boolean_step_is_not_an_integer(self):
+        with self.assertRaises(ValueError):
+            integer(False, 0, 1, "strict teacher")
+
+
+if __name__ == "__main__":
+    unittest.main()
--- /dev/null
+++ validate_pair.py
@@ -0,0 +1,301 @@
+"""ROOT ONLY stdlib byte comparator; never imports Torch or performs model math."""
+import json
+from pathlib import Path
+import signal
+import struct
+
+from support import HERE, read_json, require, sha, write
+
+PROFILES = ("official-bf16-loader", "source-36-f32-restored")
+LAYERS = (3, 7, 11, 15, 19, 23)
+TOKENS = (11, 353)
+MODELING_SHA = "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
+TORCH_SOURCE = "cf30153c4c131c8164ee7798e5022d810682e2cb"
+COUNTS = {"profiles": 2, "step_archives": 4, "canonical_domains": 168,
+          "full_vocab_logits": 4, "observed_intermediate_tensors": 60}
+STAGES = {
+    "hidden": [1, 1, 2048], "query_gate_projection": [1, 1, 4096],
+    "key_projection": [1, 1, 512], "value_projection": [1, 1, 512],
+    "query": [1, 1, 8, 256], "key": [1, 1, 2, 256],
+    "normalized_query": [1, 1, 8, 256], "normalized_key": [1, 1, 2, 256],
+    "gate": [1, 1, 2048],
+}
+
+
+def integer(value, lo, hi, description):
+    require(type(value) is int and lo <= value <= hi, description)
+    return value
+
+
+def joined(root, relative):
+    require(type(relative) is str and relative and not Path(relative).is_absolute(), "relative artifact path")
+    path = Path(root) / relative
+    require(path.resolve().is_relative_to(Path(root).resolve()) and not path.is_symlink(), "contained artifact")
+    return path
+
+
+def verified(root, row, cap=32 * 1024 * 1024):
+    path = joined(root, row["file"])
+    require(path.is_file() and 0 < path.stat().st_size <= cap and sha(path) == row["sha256"],
+            "complete original artifact SHA: " + str(path))
+    if "bytes" in row:
+        require(path.stat().st_size == row["bytes"], "recorded artifact length")
+    return path
+
+
+def archive(path):
+    path = Path(path)
+    require(path.is_file() and not path.is_symlink() and 8 < path.stat().st_size <= 32 * 1024 * 1024,
+            "bounded complete Safetensors")
+    with path.open("rb") as stream:
+        header_bytes, = struct.unpack("<Q", stream.read(8))
+        require(1 < header_bytes <= 1024 * 1024 and header_bytes + 8 <= path.stat().st_size,
+                "bounded actual Safetensors header")
+        header = json.loads(stream.read(header_bytes))
+    require(type(header) is dict, "actual tensor namespace")
+    base = header_bytes + 8
+    cursor = 0
+    spans, tensors = [], {}
+    for name, row in header.items():
+        if name == "__metadata__":
+            require(type(row) is dict and all(type(k) is str and type(v) is str for k, v in row.items()),
+                    "bounded string archive metadata")
+            continue
+        require(type(name) is str and 0 < len(name) <= 256 and type(row) is dict,
+                "actual named tensor descriptor")
+        kind, shape, offsets = row["dtype"], row["shape"], row["data_offsets"]
+        require(kind in ("BF16", "F32") and type(shape) is list and 0 < len(shape) <= 8
+                and type(offsets) is list and len(offsets) == 2, "closed raw tensor dtype/shape")
+        elements = 1
+        for width in shape:
+            elements *= integer(width, 1, 248320, "positive raw shape")
+            require(elements <= 8 * 1024 * 1024, "bounded raw tensor words")
+        lo, hi = [integer(value, 0, 32 * 1024 * 1024, "bounded tensor offset") for value in offsets]
+        require(hi - lo == elements * (2 if kind == "BF16" else 4), "exact dtype/shape byte geometry")
+        spans.append((lo, hi)); tensors[name] = {"dtype": kind, "shape": shape, "offset": base + lo, "bytes": hi - lo}
+    for lo, hi in sorted(spans):
+        require(lo == cursor, "complete nonoverlapping raw payload without gaps")
+        cursor = hi
+    require(base + cursor == path.stat().st_size and 0 < len(tensors) <= 256, "complete archive EOF/namespace")
+    return tensors
+
+
+def equal_range(left, lo, right, ro, count):
+    require(type(count) is int and count > 0, "nonvacuous byte comparison")
+    with Path(left).open("rb") as a, Path(right).open("rb") as b:
+        a.seek(lo); b.seek(ro)
+        while count:
+            width = min(count, 1024 * 1024)
+            av, bv = a.read(width), b.read(width)
+            require(len(av) == len(bv) == width and av == bv,
+                    "exact source raw bytes, including signed zero: " + str(left))
+            count -= width
+
+
+def equal_archives(left, right, count):
+    a, b = archive(left), archive(right)
+    require(set(a) == set(b) and len(a) == count, "complete equal raw tensor namespace")
+    for name in sorted(a):
+        av, bv = a[name], b[name]
+        require({key: av[key] for key in ("dtype", "shape", "bytes")}
+                == {key: bv[key] for key in ("dtype", "shape", "bytes")}, "same original tensor geometry")
+        equal_range(left, av["offset"], right, bv["offset"], av["bytes"])
+    return sum(value["bytes"] for value in a.values())
+
+
+def lifecycle(row):
+    group = row["foreground_cleanup"]
+    require(row["status"] == "CAPTURE_COMPLETED_REQUIRES_VALIDATION" and row["exit_code"] == 0
+            and row["cleanup_errors"] == [] and row["owned_container_absent_after_reap"] is True
+            and row["owned_lifetime_helper_sha256"] == "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
+            and group["all_nonzero_signals_before_reap"] is True
+            and group["kill_attempts_before_reap"] == 2 and group["stage_errors"] == []
+            and group["leader_reaped"] is True and group["group_absent"] is True
+            and group["exit_code"] == 0 and group["pgid"] == row["owned_cli_pid"]
+            and group["term_sent"] is True and group["kill_sent_before_reap"] is True
+            and row["log_cleanup"]["pipe_eof"] is True and row["log_cleanup"]["all_closed"] is True,
+            "complete exact reviewed owned lifetime")
+    for name, digest in row["entry_files"].items():
+        require(Path(name).name == name and sha(HERE / name) == digest, "original invocation source unchanged")
+
+
+def protocol(manifest, source_pins):
+    require(manifest["source_repo"] == source_pins["repo"]
+            and manifest["source_revision"] == source_pins["revision"]
+            and manifest["source_pins_sha256"] == sha(HERE / "source-pins.json")
+            and manifest["source_reverified_after_capture"] is True
+            and set(manifest["profiles"]) == set(PROFILES), "same complete source/profiles")
+    require(manifest["runtime"]["torch_num_threads"] == 4
+            and manifest["runtime"]["torch_interop_threads"] == 1
+            and manifest["runtime"]["cpu_capability"] == "AVX2"
+            and manifest["runtime"]["cuda_available"] is False, "exact accepted CPU runtime mode")
+    require(manifest["loader"]["attn_implementation"] == "eager"
+            and manifest["loader"]["use_kernels"] is False
+            and manifest["loader"]["dtype"] == "torch.bfloat16", "unchanged original official loader")
+    cfg = manifest["config"]["text_config"]
+    require(cfg["hidden_size"] == 2048 and cfg["num_attention_heads"] == 8
+            and cfg["num_key_value_heads"] == 2 and cfg["head_dim"] == 256
+            and [index for index, kind in enumerate(cfg["layer_types"]) if kind == "full_attention"] == list(LAYERS),
+            "all exact original full-attention layers")
+
+
+def observation(directory, profile, step, cumulative):
+    report = read_json(directory / "observation.json")
+    require(report["status"] == "OBSERVED_OPERATORS_ONLY_FULL_TRANSPARENCY_UNQUALIFIED"
+            and report["profile"] == profile
+            and integer(report["step"], 0, 1, "teacher step") == step
+            and integer(report["teacher_token"], 0, 248319, "teacher token") == TOKENS[step]
+            and report["modeling_sha256"] == MODELING_SHA and report["torch_source"] == TORCH_SOURCE
+            and report["no_model_or_state_or_provider_eligibility"] is True, "genuine original hook source/protocol")
+    require(report["archive"]["file"] == "attention.safetensors", "one bounded original archive")
+    path = verified(directory, report["archive"], 1024 * 1024)
+    actual = archive(path)
+    expected = {f"layer-{layer:02d}.{stage}" for layer in LAYERS for stage in STAGES}
+    require(set(actual) == set(report["tensors"]) == expected and report["tensor_count"] == 54,
+            "all six layers/nine stages exactly once")
+    total = 0
+    with path.open("rb") as stream:
+        for layer in LAYERS:
+            for stage, shape in STAGES.items():
+                name = f"layer-{layer:02d}.{stage}"
+                descriptor, meta = actual[name], report["tensors"][name]
+                require(descriptor["dtype"] == "BF16" and descriptor["shape"] == shape
+                        and meta["dtype"] == "torch.bfloat16" and meta["shape"] == shape
+                        and meta["bytes"] == descriptor["bytes"]
+                        and meta["source_module_layer"] == layer and meta["stage"] == stage,
+                        "exact source layer/stage/dtype/full geometry")
+                require(len(meta["original_stride"]) == len(meta["saved_stride"]) == len(shape)
+                        and all(type(value) is int and value > 0 for value in meta["original_stride"]),
+                        "actual nonnegative source tensor strides")
+                stride = 1
+                for width, saved in zip(reversed(shape), reversed(meta["saved_stride"])):
+                    require(type(saved) is int and saved > 0 and (width == 1 or saved == stride),
+                            "contiguous saved raw transport; singleton strides carry no payload")
+                    stride *= width
+                if stage == "query":
+                    require(meta["original_stride"][-2:] == [512, 1], "actual interleaved query/gate source view")
+                stream.seek(descriptor["offset"])
+                payload = stream.read(descriptor["bytes"])
+                require(len(payload) == descriptor["bytes"] and all((word & 0x7f80) != 0x7f80
+                        for word, in struct.iter_unpack("<H", payload)), "complete finite actual BF16 bits")
+                total += descriptor["bytes"]
+    require(report["aggregate_payload_bytes"] == cumulative + total <= 64 * 1024 * 1024,
+            "exact cumulative hook payload cap")
+    return total
+
+
+def validate(output):
+    output = Path(output)
+    parents = read_json(HERE / "parent-pins.json")
+    for pin in parents["legacy_metadata"]:
+        path = Path(pin["path"])
+        require(path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"], "immutable accepted parent evidence")
+    legacy = Path(parents["legacy_reference_root"])
+    source_pins = read_json(HERE / "source-pins.json")
+    baseline = read_json(legacy / "manifest.json")
+    accepted = read_json(legacy / "validation.json")
+    require(accepted["archive_validation"] == "PASS" and accepted["counts"] == COUNTS,
+            "actual independent accepted 2B two-profile reference")
+    protocol(baseline, source_pins)
+    totals = {}
+    for mode in ("plain", "observed"):
+        directory = output / mode
+        lifetime = read_json(directory / "invocation.json")
+        lifecycle(lifetime)
+        require(lifetime["mode"] == mode, "actual paired mode")
+        mode_report = read_json(directory / "attention-mode.json")
+        require(mode_report["mode"] == mode and mode_report["owned_hooks_removed"] is True
+                and mode_report["profiler_retired"] is True
+                and mode_report["teacher_tokens"] == list(TOKENS) and mode_report["profiles"] == list(PROFILES)
+                and mode_report["cpu_affinity"] == lifetime["expected_cpus"]
+                and len(set(mode_report["cpu_affinity"])) == 4
+                and mode_report["observer_hook_sha256"] == sha(HERE / "attention_observation_hook.py")
+                and mode_report["accepted_capture_sha256"] == parents["accepted_capture"]["sha256"]
+                and mode_report["expected_attention_tensor_count"] == (216 if mode == "observed" else 0),
+                "all actual hook/profiler owners retired and exact CPU affinity")
+        root = directory / "captured-reference"
+        validation = read_json(root / "validation.json")
+        require(validation["archive_validation"] == "PASS" and validation["counts"] == COUNTS,
+                "unchanged complete archive/layout validator passed")
+        manifest = read_json(root / "manifest.json")
+        protocol(manifest, source_pins)
+        require(manifest["capture_script_sha256"] == sha(HERE / "capture_reference.py"), "exact actual integration source")
+        for key in ("runtime_source", "package_records", "loader", "source_geometry", "tokenizer", "head", "config"):
+            require(manifest[key] == baseline[key], "unchanged original source/runtime/loader metadata: " + key)
+        raw_count = domains = intermediate_count = full_logits = attention_count = attention_bytes = compared_bytes = 0
+        for profile in PROFILES:
+            row, old = manifest["profiles"][profile], baseline["profiles"][profile]
+            require(len(row["steps"]) == len(old["steps"]) == 2, "both original teacher steps")
+            for key in ("actual_checkpoint_parameter_dtype_counts", "source_parameter_count", "source_f32_parameter_names", "restoration"):
+                require(row[key] == old[key], "unchanged profile-specific source dtype/audit")
+            audit = root / profile / "parameter-audit.json"
+            old_audit = legacy / profile / "parameter-audit.json"
+            require(sha(audit) == row["parameter_audit_sha256"] == old["parameter_audit_sha256"] == sha(old_audit),
+                    "all 632 source/617 loaded parameter audits exactly unchanged")
+            for step, (current, prior) in enumerate(zip(row["steps"], old["steps"])):
+                require(current["profile"] == prior["profile"] == profile
+                        and integer(current["step"], 0, 1, "strict actual teacher step") == prior["step"] == step
+                        and current["input_token"] == prior["input_token"] == TOKENS[step]
+                        and current["consumed_tokens"] == prior["consumed_tokens"] == list(TOKENS[:step + 1])
+                        and current["observed_official_operator_calls"] == prior["observed_official_operator_calls"],
+                        "same actual teacher and slow-GDN/conv/eager-attention call scope")
+                compared_bytes += equal_archives(verified(root, current["raw_archive"]),
+                                                  verified(legacy, prior["raw_archive"]), 49)
+                raw_count += 49; full_logits += 1
+                compared_bytes += equal_archives(verified(root, current["observed_intermediates"]),
+                                                  verified(legacy, prior["observed_intermediates"]), 15)
+                intermediate_count += 15
+                require(len(current["semantic_domains"]) == len(prior["semantic_domains"]) == 42,
+                        "all canonical domains retained")
+                actual_domains = {(value["layer"], value["kind"]): value for value in current["semantic_domains"]}
+                old_domains = {(value["layer"], value["kind"]): value for value in prior["semantic_domains"]}
+                require(len(actual_domains) == len(old_domains) == 42 and set(actual_domains) == set(old_domains),
+                        "every canonical layer/domain identity unique and complete")
+                for key, value in actual_domains.items():
+                    old_value = old_domains[key]
+                    a, b = verified(root, value), verified(legacy, old_value)
+                    require(a.stat().st_size == b.stat().st_size, "canonical domain byte geometry")
+                    equal_range(a, 0, b, 0, a.stat().st_size)
+                    domains += 1
+                attention_dir = root / profile / f"step-{step}" / "attention-observation"
+                if mode == "observed":
+                    attention_bytes += observation(attention_dir, profile, step, attention_bytes)
+                    attention_count += 54
+                else:
+                    require(not attention_dir.exists(), "plain run contains no new observation")
+        require((raw_count, domains, intermediate_count, full_logits, attention_count)
+                == (196, 168, 60, 4, 216 if mode == "observed" else 0), "nonvacuous complete paired evidence")
+        totals[mode] = {"raw_tensors_bit_exact_to_accepted": raw_count, "canonical_domains_bit_exact": domains,
+                        "prior_intermediate_tensors_bit_exact": intermediate_count, "full_vocab_logits_bit_exact": full_logits,
+                        "new_attention_tensors": attention_count, "attention_payload_bytes": attention_bytes,
+                        "raw_and_prior_intermediate_compared_bytes": compared_bytes,
+                        "manifest_sha256": sha(root / "manifest.json")}
+    result = {"status": "PASS_CPU_HF2B_PAIRED_ATTENTION_OBSERVATION_ONLY", "counts": totals,
+              "source_repo": source_pins["repo"], "source_revision": source_pins["revision"],
+              "profiles": list(PROFILES), "teacher_tokens": list(TOKENS), "stage_scope": list(STAGES),
+              "Rust_or_GPU_or_full_HF_profile_qualified": False, "matched_performance_qualified": False,
+              "scope": "fresh original CPU observer transparency to prior raw tensors plus genuine source-stage capture only"}
+    write(output / "paired-validation.json", result)
+    return result
+
+
+def bounded_validate(output):
+    def expired(_number, _frame):
+        raise TimeoutError("bounded paired byte validation exceeded120s")
+
+    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
+            "standalone validator requires no inherited alarm")
+    previous = signal.signal(signal.SIGALRM, expired)
+    signal.alarm(120)
+    try:
+        return validate(output)
+    finally:
+        signal.alarm(0)
+        signal.signal(signal.SIGALRM, previous)
+
+
+if __name__ == "__main__":
+    import argparse
+    parser = argparse.ArgumentParser()
+    parser.add_argument("directory", type=Path)
+    print(json.dumps(bounded_validate(parser.parse_args().directory), indent=2, sort_keys=True))
--- /dev/null
+++ validate_reference.py
@@ -0,0 +1,135 @@
+"""Validate archive closure and lossless layout transforms; no model execution."""
+
+import collections
+import hashlib
+import json
+import pathlib
+
+import numpy as np
+import torch
+from safetensors.torch import load_file
+
+ROOT = pathlib.Path("/output/captured-reference")
+manifest = json.loads((ROOT / "manifest.json").read_text())
+runtime_source = manifest["runtime_source"]
+pins = json.loads(pathlib.Path("/probe/source-pins.json").read_text())
+assert manifest["source_pins_sha256"] == hashlib.sha256(pathlib.Path("/probe/source-pins.json").read_bytes()).hexdigest()
+for pin in runtime_source["files"].values():
+    for path in [pathlib.Path(pin["runtime_path"])]:
+        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"], path
+assert manifest["official_modeling_source"]["sha256"] == runtime_source["files"]["transformers/models/qwen3_5/modeling_qwen3_5.py"]["sha256"]
+assert manifest["source_reverified_after_capture"]
+assert manifest["limits"]["container_memory_max"] == str(12 * 1024**3)
+assert manifest["limits"]["container_cpu_max"] == "400000 100000"
+assert not manifest["runtime"]["cuda_available"]
+assert manifest["runtime"]["torch_num_threads"] == 4
+profiles = ["official-bf16-loader", "source-36-f32-restored"]
+counts = {"profiles": 0, "step_archives": 0, "canonical_domains": 0, "full_vocab_logits": 0, "observed_intermediate_tensors": 0}
+
+
+def verified_file(row):
+    path = ROOT / row["file"]
+    assert path.resolve().is_relative_to(ROOT) and not path.is_symlink() and path.is_file()
+    assert path.stat().st_size <= 32 * 1024**2
+    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
+    return path
+
+
+def verify_f32(row, expected):
+    actual = np.fromfile(verified_file(row), dtype="<f4")
+    expected = expected.detach().cpu().float().contiguous().numpy().reshape(-1)
+    # Compare bits, including zero sign; array_equal would hide that distinction.
+    assert np.array_equal(actual.view("<u4"), expected.view("<u4"))
+    assert actual.size == row["numel"] and np.isfinite(actual).all()
+
+
+for profile in profiles:
+    data = manifest["profiles"][profile]
+    audit_path = ROOT / profile / "parameter-audit.json"
+    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == data["parameter_audit_sha256"]
+    audit = json.loads(audit_path.read_text())
+    assert len(audit) == 632
+    assert all(row["exact_after_declared_dtype_cast"] for row in audit if row["loaded"] is not None)
+    excluded = [row for row in audit if row["loaded"] is None]
+    assert len(excluded) == 15 and all(row["name"].startswith("mtp.") for row in excluded)
+    original_f32 = [row for row in audit if row["source"]["dtype"] == "torch.float32"]
+    assert len(original_f32) == 36
+    assert [row["name"] for row in original_f32] == pins["raw_f32_names"]
+    assert dict(collections.Counter(row["loaded"]["dtype"] for row in audit if row["loaded"] is not None)) == data["actual_checkpoint_parameter_dtype_counts"]
+    assert len(data["steps"]) == 2
+    if profile == "source-36-f32-restored":
+        assert all(row["loaded"]["dtype"] == "torch.float32" for row in original_f32)
+    counts["profiles"] += 1
+    for step, saved in enumerate(data["steps"]):
+        raw = load_file(str(verified_file(saved["raw_archive"])), device="cpu")
+        assert len(raw) == 49
+        assert raw["logits"].shape == (1, 1, 248320)
+        assert raw["logits"].dtype == torch.bfloat16
+        assert saved["consumed_tokens"] == [11, 353][:step+1] and saved["step"] == step
+        assert saved["semantic_domain_count"] == 42 and len(saved["raw_tensor_layouts"]) == 49
+        for layer in range(24):
+            prefix = f"layer-{layer:02d}"
+            if layer % 4 == 3:
+                for key in ["keys", "values"]:
+                    assert raw[prefix + "." + key].shape == (1, 2, step+1, 256)
+                    assert raw[prefix + "." + key].dtype == torch.bfloat16
+            else:
+                assert raw[prefix + ".conv"].shape == (1, 6144, 4)
+                assert raw[prefix + ".conv"].dtype == torch.bfloat16
+                assert raw[prefix + ".recurrent"].shape == (1, 16, 128, 128)
+                assert raw[prefix + ".recurrent"].dtype == torch.float32
+        assert len(saved["semantic_domains"]) == 42
+        observed = load_file(str(verified_file(saved["observed_intermediates"])), device="cpu")
+        assert len(observed) == 15
+        assert torch.equal(observed["final_logits"].contiguous().view(torch.uint8), raw["logits"].contiguous().view(torch.uint8))
+        assert torch.equal(observed["layer-00.gdn_recurrent_fp32"].contiguous().view(torch.uint8), raw["layer-00.recurrent"].permute(0, 1, 3, 2).contiguous().view(torch.uint8))
+        assert all(bool(torch.isfinite(value).all()) for value in observed.values())
+        counts["observed_intermediate_tensors"] += len(observed)
+        verify_f32(saved["logits_f32"], raw["logits"])
+        counts["full_vocab_logits"] += 1
+        expected_domains = {(layer, kind) for layer in range(24) for kind in
+                            (["attention-kv-pair"] if layer % 4 == 3 else
+                             ["gdn-convolution-history", "gdn-recurrent"])}
+        seen = set()
+        for domain in saved["semantic_domains"]:
+            key = (domain["layer"], domain["kind"])
+            assert key in expected_domains and key not in seen
+            seen.add(key)
+            prefix = f"layer-{domain['layer']:02d}"
+            if domain["kind"] == "gdn-convolution-history":
+                expected = raw[prefix + ".conv"][..., -3:]
+            elif domain["kind"] == "gdn-recurrent":
+                expected = raw[prefix + ".recurrent"].permute(0, 1, 3, 2)
+            else:
+                key = raw[prefix + ".keys"].permute(0, 2, 1, 3).contiguous().reshape(-1)
+                val = raw[prefix + ".values"].permute(0, 2, 1, 3).contiguous().reshape(-1)
+                expected = torch.cat([key, val])
+            verify_f32(domain, expected)
+            counts["canonical_domains"] += 1
+        assert seen == expected_domains
+        assert all(bool(torch.isfinite(value).all()) for value in raw.values())
+        counts["step_archives"] += 1
+
+differences = []
+for step in range(2):
+    left_data = manifest["profiles"][profiles[0]]["steps"][step]
+    right_data = manifest["profiles"][profiles[1]]["steps"][step]
+    left = load_file(str(ROOT / left_data["raw_archive"]["file"]))
+    right = load_file(str(ROOT / right_data["raw_archive"]["file"]))
+    rows = []
+    for name in sorted(left):
+        a, b = left[name].float().contiguous().numpy(), right[name].float().contiguous().numpy()
+        changed = int(np.count_nonzero(a.view("<u4") != b.view("<u4")))
+        rows.append({"tensor": name, "elements": int(a.size), "bit_different_elements": changed, "max_abs_delta": float(np.abs(a - b).max())})
+    differences.append({"step": step, "tensors": rows})
+result = {"archive_validation": "PASS", "counts": counts, "runtime_source_files_verified": len(runtime_source["files"]), "model_or_math_parity_qualified": False, "source_revision": pins["revision"], "cross_profile_differences": differences, "scope": "archive closure, full finite tensors and exact layout derivations only; numerical profile differences are reported without a tolerance or cross-engine acceptance claim"}
+total_artifacts = 0
+for path in ROOT.rglob("*"):
+    assert not path.is_symlink()
+    if path.is_file():
+        total_artifacts += path.stat().st_size
+assert total_artifacts <= 256 * 1024**2
+assert 0 < int(manifest["cgroup_memory_peak_bytes"]) <= 12 * 1024**3
+assert counts == {"profiles": 2, "step_archives": 4, "canonical_domains": 168, "full_vocab_logits": 4, "observed_intermediate_tensors": 60}
+(ROOT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
+print(json.dumps({"archive_validation": "PASS", "counts": counts}, sort_keys=True))

````

## Source: .agents/runs/model-execution-hf2b-cpu-reference-20261020/LIFECYCLE-NAME-FOLLOWUP.patch

Original UTF-8 bytes: 875; SHA256 `1492aba1908d96a40b1088e605772fac8fe034f13b702d811c1330007ca0d713`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/run_probe.py
+++ b/run_probe.py
@@ -190,8 +190,8 @@
                                             'full_vocab_logits': 4, 'observed_intermediate_tensors': 60}
                 record['archive_validation_passed'] = True
                 record['counts'] = report['counts']
-                for name, expected in record['scripts'].items():
-                    assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == expected, 'script changed'
+                for script_name, expected in record['scripts'].items():
+                    assert hashlib.sha256((RUN / script_name).read_bytes()).hexdigest() == expected, 'script changed'
                 assert hashlib.sha256((RUN / 'source-pins.json').read_bytes()).hexdigest() == record['source_pins_sha256']
         except BaseException as error:
             record['primary_failure'] = traceback.format_exc()

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 91410; SHA256 `3a853fb4b7b5d6e088bafef486d4259f4ad519bb5a5a9a6d4a6a23cb014bf181`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/.agents/runs/model-execution-original-attention-replay-20261020/Cargo.toml b/.agents/runs/model-execution-original-attention-replay-20261020/Cargo.toml
new file mode 100644
--- /dev/null
+++ b/.agents/runs/model-execution-original-attention-replay-20261020/Cargo.toml
@@ -0,0 +1,23 @@
+[package]
+name = "original-attention-fragment-replay"
+version = "0.0.0"
+edition = "2024"
+publish = false
+
+[workspace]
+
+[dependencies]
+zerogravity-artifacts = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-artifacts", features = ["cache"] }
+zerogravity-core = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core" }
+zerogravity-model-qwen35 = { path = "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35", features = ["cpu", "original-hf-binding"] }
+serde_json = "=1.0.151"
+sha2 = "=0.10.9"
+libc = "=0.2.190"
+
+[profile.release]
+opt-level = 3
+codegen-units = 1
+
+[[bin]]
+name = "original-attention-fragment-replay"
+path = "main.rs"
diff --git a/.agents/runs/model-execution-original-attention-replay-20261020/README.md b/.agents/runs/model-execution-original-attention-replay-20261020/README.md
new file mode 100644
--- /dev/null
+++ b/.agents/runs/model-execution-original-attention-replay-20261020/README.md
@@ -0,0 +1,94 @@
+# Root-only authentic CPU attention fragment replay
+
+- Parent: existing model-execution plan. Source preparation only; root executes.
+- Current MAIN APIs only: sealed original `VerifiedTensorSource` →
+  `OriginalWindowOperators::bind/execute_attention_projection_norm_slice` plus
+  explicit `AttentionSliceInput`. Same shared HostMath/ForwardPlan operations;
+  no alternate mathematical implementation, token loop, state or input node.
+- Sources: original .8B@2fc063… and 2B@15852…; raw original files/headers/source
+  identity, BF16 matrices/learned QK gamma and epsilon1e-6 stay intact. All320
+  text-role descriptors bind. Only `source-36-f32-restored` CPU AVX2 semantics;
+  official all-BF16 loader and CUDA/vLLM mixed cast profile remain distinct.
+
+## Independent reference and exact scope
+
+- Two genuine prior teacher inputs,11 then353, at first FullAttention layer3.
+  Pure fragments receive already-observed authentic hidden inputs; they do not
+  produce those inputs or replay the preceding model/cache computation.
+- .8B accepted root-probe-2: immutable whole archive pins plus actual call001
+  original public schemas, full source parameters, input byte digests, split
+  query frame/gate-flat frame and final type_as BF16 norm outputs. Parent all98
+  logits/cache checks,12attention+2rotary and2802public checks stay mandatory.
+- 2B accepted root-pair-1: e45e1903… paired validation/f969e1a4… observed manifest,
+  both plain/observed196raw tensors/four fullV/168domains/60prior intermediates
+  exact to accepted reference; raw source public-hook stage/layer/strides.
+  No additional Torch/operator/model/container or D2H operation is required.
+- Exporter copies only original raw BF16 words: four inputs plus seven expected
+  outputs per case =106,496B. Expected tensors are never reconstructed from
+  projections, rounded, zero-normalized or fitted. Independent validator
+  rechecks exported words against the original authenticated archive spans.
+- Rust compares all47,104 output words (23,552/source) as raw BF16 widened bits,
+  including signed zero. Seven outputs: query/gate projection,K projection,V
+  projection,query,gate,normalized query,normalized key. Full lengths required.
+  Actual raw BF16 output files permit independent stdlib byte comparison.
+
+## Real ownership and negative gates
+
+- Same original full-shard/index FD/`prepare_verified_windows/open` path as the
+  accepted root window factory; no materialized 1.7/4.5GB learned cache or fake
+  reduced descriptors. Root performs complete source-file scans before/after;
+  original window open independently authenticates the full weight shard.
+- One128MiB host ResourceArbiter domain per sequential source:64MiB admitted
+  source metadata,4MiB binding metadata,2MiB harness control/reference envelope.
+  Existing per-value64KiB Output+960KiB Scratch retain original source issuer.
+  Each case retains eight backings (input+seven outputs),16MiB for both cases.
+- Actual pool exhausted through same-source workspace leases; execution must
+  refuse and release partial private work. Pre-cancel and wrong GDN layer also
+  refuse. Mid-cancel is triggered by actual >=2 Output backings plus a real
+  TransferStaging charge. Missing this short physical boundary fails the case;
+  it cannot be reported as passed merely because a timer elapsed.
+- Both outputs survive dropping callers/operators/source, are compared again,
+  then release all source and backing owners. Final reservations/bytes0 and
+  no fence required. No SessionAuthority/StateRoot is allocated or published.
+- Pool accounting covers named payloads/explicit metadata. Generic allocator,
+  executable/JSON runtime bookkeeping, OS caches and RSS do not become physical
+  free-RAM admission claims. No event-complete peak measurement is claimed.
+- Native launcher uses the exact accepted root resolver WNOWAIT/owned SID/PGID
+  cleanup functions. Independent TERM/grace/KILL stages occur before any reap;
+  final reap, group absence, bounded pipe drain and closure required. Both
+  helper invocations are serialized and use the same pinned executable FD.
+
+## ROOT execution after independent source review
+
+All commands are prepared, **unexecuted by author**. Keep old accepted archives,
+source observations and prior attention proposal bytes immutable.
+
+```sh
+cd /home/pvl/spaces/abird/src/zerogravity
+RUN="$PWD/.agents/runs/model-execution-original-attention-replay-20261020"
+# Source-free parser/transport fixtures; no trained qualification.
+PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$RUN" -p test_contracts.py -v
+# Genuine existing archive transport only; fresh directory directly under RUN.
+PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/prepare_inputs.py" --new-output "$RUN/root-inputs-1"
+# Root records the actual Cargo argv, source freeze and fresh binary SHA.
+CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
+  cargo build --offline --release --manifest-path "$RUN/Cargo.toml"
+# Actual helper/inputs SHA values are obtained by ROOT, never placeholder pins.
+python3 scripts/native-gpu.py -- python3 "$RUN/run_replay.py" \
+  --binary ROOT_FRESH_BINARY --binary-sha256 ROOT_ACTUAL_SHA \
+  --inputs "$RUN/root-inputs-1/inputs.json" --inputs-sha256 ROOT_INPUTS_SHA \
+  --new-output "$RUN/root-replay-1"
+PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/validate_replay.py" --output "$RUN/root-replay-1"
+```
+
+Root must retain actual build stdout/argv and source/binary attestation; the
+launcher cannot prove compilation provenance merely from a supplied SHA. Its
+freeze binds the exact current MAIN source bytes and API dependencies. Cargo
+lock normalization follows the accepted root helper policy; inspect any delta.
+No dependency fetch or SDK/model acquisition is needed. Root schedules complete
+source scans outside benchmarks and competing model/GPU phases.
+
+Possible scope after all checks: **original CPU private attention fragment only**.
+RoPE,KV/cache append,BMM/softmax/gating integration,initial/cached GDN coefficients
+and MKL scheduling,full-HF session/checkpoint/logits,27B,GPU/native cuBLAS and
+matched serving performance remain unqualified.
diff --git a/.agents/runs/model-execution-original-attention-replay-20261020/TODO.md b/.agents/runs/model-execution-original-attention-replay-20261020/TODO.md
new file mode 100644
--- /dev/null
+++ b/.agents/runs/model-execution-original-attention-replay-20261020/TODO.md
@@ -0,0 +1,15 @@
+# Working checklist
+
+- Parent: existing model-execution plan; owner batch_design_review.
+- Owned scope: this new ignored run only. MAIN source/shared plans unchanged.
+- [x] Review narrow current MAIN attention compile/shape fixes; no blocker.
+- [x] Prepare same-engine authentic .8B/2B layer3 two-teacher seven-output harness.
+- [x] Keep genuine public source/schema/profile/archive and parent transparency binding.
+- [x] Prepare real same-source pool pressure,cancel,retained owner/final0 checks.
+- [x] Reuse exact accepted owned-native lifecycle functions; preserve original.
+- [ ] Independent source review,source-free fixtures,actual archive export.
+- [ ] ROOT build/current source+fresh binary binding and exact47,104-word replay.
+- [ ] ROOT strict/full phase checks and acceptance; no implementation promotion here.
+
+Author: AST/source inspection only. No tests,compilation,model/operator/GPU or
+container execution; no large raw/model/DSO read/hash. No physical result claimed.
diff --git a/.agents/runs/model-execution-original-attention-replay-20261020/bindings.json b/.agents/runs/model-execution-original-attention-replay-20261020/bindings.json
new file mode 100644
--- /dev/null
+++ b/.agents/runs/model-execution-original-attention-replay-20261020/bindings.json
@@ -0,0 +1,272 @@
+{
+  "layer": 3,
+  "models": {
+    "0.8b": {
+      "hidden": 1024,
+      "invocation": {
+        "bytes": 3522,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-attention-observation-20261020/root-probe-2/invocation.json",
+        "sha256": "7b55a459f01f89af079a607260e8a14bad1fcf7f3f8942be559d5c7b0603ac0d"
+      },
+      "kv_heads": 2,
+      "query_heads": 8,
+      "repo": "Qwen/Qwen3.5-0.8B",
+      "report": {
+        "bytes": 1141416,
+        "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-attention-observation-20261020/root-probe-2/report.json",
+        "sha256": "b485f40cfb31bbc2fd3196ac8ae5686268974b74161df944d914a762cda8aa33"
+      },
+      "revision": "2fc06364715b967f1860aea9cf38778875588b17",
+      "source_directory": "/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17",
+      "source_files": [
+        {
+          "bytes": 11544,
+          "path": "LICENSE",
+          "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
+        },
+        {
+          "bytes": 7755,
+          "path": "chat_template.jinja",
+          "sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
+        },
+        {
+          "bytes": 2907,
+          "path": "config.json",
+          "sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
+        },
+        {
+          "bytes": 3353259,
+          "path": "merges.txt",
+          "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
+        },
+        {
+          "bytes": 1746942600,
+          "path": "model.safetensors-00001-of-00001.safetensors",
+          "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
+        },
+        {
+          "bytes": 50900,
+          "path": "model.safetensors.index.json",
+          "sha256": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e"
+        },
+        {
+          "bytes": 390,
+          "path": "preprocessor_config.json",
+          "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
+        },
+        {
+          "bytes": 12807982,
+          "path": "tokenizer.json",
+          "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
+        },
+        {
+          "bytes": 16709,
+          "path": "tokenizer_config.json",
+          "sha256": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"
+        },
+        {
+          "bytes": 385,
+          "path": "video_preprocessor_config.json",
+          "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
+        },
+        {
+          "bytes": 6722759,
+          "path": "vocab.json",
+          "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
+        }
+      ],
+      "steps": [
+        {
+          "archive": {
+            "bytes": 2697148,
+            "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-attention-observation-20261020/root-probe-2/step-0/attention.safetensors",
+            "sha256": "bdc827153bc1a7c667146b2831bb22840b7ad626ecd5a653888c95b33150e34a"
+          },
+          "metadata": {
+            "bytes": 818673,
+            "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-attention-observation-20261020/root-probe-2/step-0/validated-original.json",
+            "sha256": "bf4b16000a782727c58995589d50786d29ada54f1cde827a5ea3a24bb0dd525e"
+          }
+        },
+        {
+          "archive": {
+            "bytes": 3089820,
+            "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-attention-observation-20261020/root-probe-2/step-1/attention.safetensors",
+            "sha256": "fdf634861d2bb8a6c8aa0e41385c25567e2cb33de2077aa681f8e498306d7928"
+          },
+          "metadata": {
+            "bytes": 802107,
+            "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-attention-observation-20261020/root-probe-2/step-1/validated-original.json",
+            "sha256": "a8cf82eaa22f52b40a99b05d5ca3e5669fff94808f9c052f6d65cad963bd5391"
+          }
+        }
+      ]
+    },
+    "2b": {
+      "accepted_validation": {
+        "bytes": 17355,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/root-pair-1/observed/captured-reference/validation.json",
+        "sha256": "6d7d6ba1a16555c4930c922c3bcd6a6c8d957e6c231b3d9b5b6924dd54a3ee92"
+      },
+      "hidden": 2048,
+      "kv_heads": 2,
+      "manifest": {
+        "bytes": 313368,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/root-pair-1/observed/captured-reference/manifest.json",
+        "sha256": "f969e1a42281a33883c713c8fa96bf50ab77047751e44ab74d3bd1597c2054e0"
+      },
+      "observed_invocation": {
+        "bytes": 4198,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/root-pair-1/observed/invocation.json",
+        "sha256": "e75d4b76d35d75bd00a4c818afae10261bf0865b5d9895e1a7849c8d5eaf2c6a"
+      },
+      "paired": {
+        "bytes": 1570,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/root-pair-1/paired-validation.json",
+        "sha256": "e45e190356b7de64d205113a4b7cfc02ff4b4a745c2e878da12ef53ed2d9767e"
+      },
+      "plain_invocation": {
+        "bytes": 4179,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/root-pair-1/plain/invocation.json",
+        "sha256": "c8f4e23fd9a79f7eb1ecdc711aa1d1f08ecfc6e4c3e600d8c4b8e2be1de3e5e8"
+      },
+      "query_heads": 8,
+      "repo": "Qwen/Qwen3.5-2B",
+      "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
+      "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52/files",
+      "source_files": [
+        {
+          "bytes": 11544,
+          "path": "LICENSE",
+          "provenance": "independent official Git blob SHA1 plus cache SHA256",
+          "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
+        },
+        {
+          "bytes": 62814,
+          "path": "README.md",
+          "provenance": "independent official Git blob SHA1 plus cache SHA256",
+          "sha256": "c0e83a849c776e6fa843d011f023132d942a0f0140d903205bf1c363adad2275"
+        },
+        {
+          "bytes": 7755,
+          "path": "chat_template.jinja",
+          "provenance": "independent official Git blob SHA1 plus cache SHA256",
+          "sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
+        },
+        {
+          "bytes": 2908,
+          "path": "config.json",
+          "provenance": "independent official Git blob SHA1 plus cache SHA256",
+          "sha256": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"
+        },
+        {
+          "bytes": 3353259,
+          "path": "merges.txt",
+          "provenance": "independent official Git blob SHA1 plus cache SHA256",
+          "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
+        },
+        {
+          "bytes": 4548221488,
+          "path": "model.safetensors-00001-of-00001.safetensors",
+          "provenance": "independent official LFS SHA256",
+          "sha256": "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
+        },
+        {
+          "bytes": 64460,
+          "path": "model.safetensors.index.json",
+          "provenance": "independent official Git blob SHA1 plus cache SHA256",
+          "sha256": "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9"
+        },
+        {
+          "bytes": 390,
+          "path": "preprocessor_config.json",
+          "provenance": "independent official Git blob SHA1 plus cache SHA256",
+          "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
+        },
+        {
+          "bytes": 12807982,
+          "path": "tokenizer.json",
+          "provenance": "independent official LFS SHA256",
+          "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
+        },
+        {
+          "bytes": 16709,
+          "path": "tokenizer_config.json",
+          "provenance": "independent official Git blob SHA1 plus cache SHA256",
+          "sha256": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"
+        },
+        {
+          "bytes": 385,
+          "path": "video_preprocessor_config.json",
+          "provenance": "independent official Git blob SHA1 plus cache SHA256",
+          "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
+        },
+        {
+          "bytes": 6722759,
+          "path": "vocab.json",
+          "provenance": "independent official Git blob SHA1 plus cache SHA256",
+          "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
+        }
+      ],
+      "source_pins": {
+        "bytes": 10079,
+        "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/source-pins.json",
+        "sha256": "9bd17bb4d58582373034fd7b0097e36d3e39bf91d22bce837cb7fe109ee9fee4"
+      },
+      "steps": [
+        {
+          "archive": {
+            "bytes": 177008,
+            "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/root-pair-1/observed/captured-reference/source-36-f32-restored/step-0/attention-observation/attention.safetensors",
+            "sha256": "8ebf49c3debe2c692ab77d02dc40919be92333bc588b001d7312f4ed2166a231"
+          },
+          "metadata": {
+            "bytes": 20712,
+            "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/root-pair-1/observed/captured-reference/source-36-f32-restored/step-0/attention-observation/observation.json",
+            "sha256": "53fcdee8cf24870e20b49c95b9d22163d0a3b67dc70afd34a212e2a35cbb3181"
+          }
+        },
+        {
+          "archive": {
+            "bytes": 177008,
+            "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/root-pair-1/observed/captured-reference/source-36-f32-restored/step-1/attention-observation/attention.safetensors",
+            "sha256": "bd784e6c5305e315d67b70914ce978b060975b75b654e7d36e16991f5c534775"
+          },
+          "metadata": {
+            "bytes": 20713,
+            "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/root-pair-1/observed/captured-reference/source-36-f32-restored/step-1/attention-observation/observation.json",
+            "sha256": "0a7efd7ea966edc8d4921f7a89ed059f125ae72b862b9e9453c99db686af3d40"
+          }
+        }
+      ]
+    }
+  },
+  "profile": "source-36-f32-restored",
+  "protocol_sources": [
+    {
+      "bytes": 26729,
+      "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-attention-observation-20261020/executed-root-probe-2-scripts/observe_and_replay.py",
+      "sha256": "fc5d283b903e9802cc575e816eefb921da08295d33f2d4640cb23fde615fe96f"
+    },
+    {
+      "bytes": 9970,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/attention_observation_hook.py",
+      "sha256": "12407465ee6795c2c94d191f1ff3d5550b7bef29e8d916f8f824b6b736dd9495"
+    },
+    {
+      "bytes": 18104,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/validate_pair.py",
+      "sha256": "30cfeb15ba809dc14bb2814d1f70d0b9a262579120e3b4cc8ddf54ec0e2de279"
+    },
+    {
+      "bytes": 3438,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020/freeze-manifest.json",
+      "sha256": "853f002958a1a5a1b21493376203b10a12180700b74bfdcc26f9f435def9783a"
+    }
+  ],
+  "schema": "zerogravity.original-attention-fragment.bindings.v1",
+  "teacher_tokens": [
+    11,
+    353
+  ]
+}
diff --git a/.agents/runs/model-execution-original-attention-replay-20261020/contracts.py b/.agents/runs/model-execution-original-attention-replay-20261020/contracts.py
new file mode 100644
--- /dev/null
+++ b/.agents/runs/model-execution-original-attention-replay-20261020/contracts.py
@@ -0,0 +1,185 @@
+"""Stdlib transport/contracts only; importing performs no IO or math/runtime work."""
+import hashlib
+import json
+from pathlib import Path
+import struct
+
+HERE = Path(__file__).resolve().parent
+PROFILE = "source-36-f32-restored"
+TOKENS = (11, 353)
+MODELING = "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
+TORCH_GIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
+STAGES = ("query_gate_projection", "key_projection", "value_projection", "query", "gate",
+          "normalized_query", "normalized_key")
+SHAPES = {"query_gate_projection": [1, 1, 4096], "key_projection": [1, 1, 512],
+          "value_projection": [1, 1, 512], "query": [1, 1, 8, 256],
+          "gate": [1, 1, 2048], "normalized_query": [1, 1, 8, 256],
+          "normalized_key": [1, 1, 2, 256]}
+LABELS_8 = {"hidden": "call001.hidden", "query_gate_projection": "call001.op000.result",
+            "key_projection": "call001.op016.result", "value_projection": "call001.op029.result",
+            "query": "call001.frame.forward.query_gate_split.query_states",
+            "gate": "call001.frame.forward.gate_flat.gate",
+            "normalized_query": "call001.op014.result", "normalized_key": "call001.op027.result"}
+CAP = 32 * 1024 * 1024
+
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+
+def sha(path):
+    with Path(path).open("rb") as opened:
+        return hashlib.file_digest(opened, "sha256").hexdigest()
+
+
+def pin(path, row, cap=CAP):
+    path = Path(path)
+    require(not path.is_symlink() and path.is_file() and 0 < path.stat().st_size <= cap,
+            "bounded regular pinned artifact: " + str(path))
+    require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
+            "immutable artifact differs: " + str(path))
+    return path
+
+
+def read_json(path, cap=2 * 1024 * 1024):
+    path = Path(path)
+    require(not path.is_symlink() and path.is_file() and 0 < path.stat().st_size <= cap,
+            "bounded JSON artifact")
+    def unique(pairs):
+        result = {}
+        for key, value in pairs:
+            require(key not in result, "duplicate JSON field")
+            result[key] = value
+        return result
+    return json.loads(path.read_text(), object_pairs_hook=unique)
+
+
+def write(path, value):
+    with Path(path).open("x", encoding="utf-8") as opened:
+        json.dump(value, opened, indent=2, sort_keys=True)
+        opened.write("\n")
+
+
+def relative(value):
+    require(type(value) is str and value and not Path(value).is_absolute()
+            and all(part not in (".", "..") for part in Path(value).parts), "contained relative file")
+    return value
+
+
+def finite(raw, words):
+    require(len(raw) == words * 2 and words > 0 and all((word & 0x7f80) != 0x7f80
+            for word, in struct.iter_unpack("<H", raw)), "exact finite full BF16 payload")
+
+
+def archive(path):
+    """Only descriptors: exact nonoverlapping whole archive, never tensor arithmetic."""
+    path = Path(path)
+    require(not path.is_symlink() and 8 < path.stat().st_size <= CAP, "bounded Safetensors")
+    with path.open("rb") as opened:
+        count, = struct.unpack("<Q", opened.read(8))
+        require(1 < count <= 1024 * 1024 and count + 8 <= path.stat().st_size, "bounded tensor header")
+        raw = opened.read(count)
+    def unique(pairs):
+        result = {}
+        for key, value in pairs:
+            require(key not in result, "duplicate tensor/descriptor field")
+            result[key] = value
+        return result
+    header = json.loads(raw, object_pairs_hook=unique)
+    tensors, spans = {}, []
+    for name, descriptor in header.items():
+        if name == "__metadata__":
+            require(type(descriptor) is dict, "string archive metadata")
+            continue
+        dtype, shape = descriptor["dtype"], descriptor["shape"]
+        require(type(name) is str and len(name) <= 256 and dtype in ("BF16", "F32", "F64", "I64", "I32", "BOOL")
+                and type(shape) is list and len(shape) <= 8, "closed tensor descriptor")
+        size = {"BF16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "BOOL": 1}[dtype]
+        for width in shape:
+            require(type(width) is int and 0 <= width <= 248320, "bounded source axis, including observed empty tensors")
+            size *= width
+            require(size <= CAP, "bounded source tensor")
+        lo, hi = descriptor["data_offsets"]
+        require(type(lo) is type(hi) is int and 0 <= lo <= hi <= CAP and hi - lo == size,
+                "exact tensor byte geometry")
+        tensors[name] = {"dtype": dtype, "shape": shape, "offset": 8 + count + lo, "bytes": size}
+        spans.append((lo, hi))
+    cursor = 0
+    for lo, hi in sorted(spans):
+        require(lo == cursor, "complete disjoint tensor payload")
+        cursor = hi
+    require(8 + count + cursor == path.stat().st_size and 0 < len(tensors) <= 4096,
+            "complete archive EOF and namespace")
+    return tensors
+
+
+def payload(path, descriptor, shape):
+    require(descriptor["dtype"] == "BF16" and descriptor["shape"] == shape, "actual BF16 stage shape")
+    with Path(path).open("rb") as opened:
+        opened.seek(descriptor["offset"])
+        raw = opened.read(descriptor["bytes"])
+    finite(raw, descriptor["bytes"] // 2)
+    return raw
+
+
+def source_8(report, step, metadata):
+    require(report["source_public_oracle_passed"] is True and report["rust_qualified"] is False
+            and report["actual_attention_calls"] == 12 and report["actual_rotary_calls"] == 2
+            and report["prior_checks"] == 98 and len(report["replay_checks"]) == 2802
+            and report["modeling_sha256"] == MODELING and report["runtime"]["torch_git"] == TORCH_GIT
+            and report["runtime"]["cpu"] == "AVX2" and report["runtime"]["threads"] == 4
+            and report["runtime"]["interop_threads"] == 1
+            and report["source_receipt"]["repo"] == "Qwen/Qwen3.5-0.8B"
+            and report["source_receipt"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
+            and report["steps"][step]["token"] == TOKENS[step]
+            and metadata["prior_checks"] == 49, "actual accepted .8B source/runtime/complete transparency")
+    call = metadata["calls"][1]
+    require(call["kind"] == "attention" and call["layer"] == 3 and call["prefix"] == "call001"
+            and call["hidden"] == LABELS_8["hidden"] and call["inference_mode"] is True,
+            "actual first FullAttention frame")
+    for index, member, rows in ((0, "q_proj", 4096), (16, "k_proj", 512), (29, "v_proj", 512)):
+        op = call["operations"][index]
+        parameter = op["args"]["tuple_items"][1]["source_parameter"]
+        require(op["operator"] == "aten.linear.default"
+                and op["schema"] == "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"
+                and op["result"] == {"tensor_key": f"call001.op{index:03d}.result"}
+                and parameter == {"name": f"model.language_model.layers.3.self_attn.{member}.weight",
+                                  "source_dtype": "torch.bfloat16", "source_shape": [rows, 1024]},
+                "genuine public linear schema/original full parameter")
+        input_key = op["args"]["tuple_items"][0]["tensor_key"]
+        require(metadata["tensors"][input_key]["sha256"] == metadata["tensors"][LABELS_8["hidden"]]["sha256"]
+                and metadata["tensors"][input_key]["shape"] == [1, 1, 1024],
+                "all three original linear calls consume the same authentic attention input")
+    for index in (14, 27):
+        require(call["operations"][index]["operator"] == "aten.type_as.default", "actual final Q/K norm BF16 cast")
+    require(call["frames"]["forward.query_gate_split.query_states"] == LABELS_8["query"]
+            and call["frames"]["forward.gate_flat.gate"] == LABELS_8["gate"], "actual split/flat source frame labels")
+
+
+def source_2(paired, manifest, step, observation):
+    require(paired["status"] == "PASS_CPU_HF2B_PAIRED_ATTENTION_OBSERVATION_ONLY"
+            and paired["source_revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
+            and paired["teacher_tokens"] == list(TOKENS) and PROFILE in paired["profiles"], "accepted 2B paired reference")
+    for mode in ("plain", "observed"):
+        row = paired["counts"][mode]
+        require(row["raw_tensors_bit_exact_to_accepted"] == 196
+                and row["full_vocab_logits_bit_exact"] == 4 and row["canonical_domains_bit_exact"] == 168
+                and row["prior_intermediate_tensors_bit_exact"] == 60, "paired source/raw/fullV/all-cache transparency")
+    require(manifest["source_repo"] == "Qwen/Qwen3.5-2B" and manifest["source_revision"] == paired["source_revision"]
+            and manifest["source_reverified_after_capture"] is True
+            and manifest["loader"]["attn_implementation"] == "eager" and manifest["loader"]["use_kernels"] is False
+            and manifest["runtime"]["cpu_capability"] == "AVX2"
+            and manifest["runtime"]["torch_num_threads"] == 4
+            and manifest["runtime"]["cuda_available"] is False
+            and manifest["official_modeling_source"]["sha256"] == MODELING
+            and manifest["profiles"][PROFILE]["profile"] == PROFILE
+            and manifest["profiles"][PROFILE]["actual_checkpoint_parameter_dtype_counts"]["torch.float32"] == 36
+            and len(manifest["profiles"][PROFILE]["source_f32_parameter_names"]) == 36
+            and manifest["profiles"][PROFILE]["steps"][step]["input_token"] == TOKENS[step],
+            "unchanged accepted original 2B restored-source CPU loader/protocol")
+    require(observation["profile"] == PROFILE and observation["step"] == step
+            and observation["teacher_token"] == TOKENS[step] and observation["modeling_sha256"] == MODELING
+            and observation["torch_source"] == TORCH_GIT and observation["tensor_count"] == 54
+            and observation["no_model_or_state_or_provider_eligibility"] is True,
+            "genuine 2B original public hook/profiler binding")
diff --git a/.agents/runs/model-execution-original-attention-replay-20261020/lifecycle-provenance.json b/.agents/runs/model-execution-original-attention-replay-20261020/lifecycle-provenance.json
new file mode 100644
--- /dev/null
+++ b/.agents/runs/model-execution-original-attention-replay-20261020/lifecycle-provenance.json
@@ -0,0 +1,11 @@
+{
+  "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-qwen-window-resolver-20261020/.agents/runs/model-execution-qwen-window-resolver-20261020/actual-replay/run_replay_v2.py",
+  "bytes": 9862,
+  "sha256": "2260539526a917b1d2f878a536238d3b5705dfc80f7f4c93e6a1b27188f10dfa",
+  "functions_extracted_verbatim": [
+    "cleanup_owned",
+    "launch_limits",
+    "unreaped"
+  ],
+  "author_executed": false
+}
diff --git a/.agents/runs/model-execution-original-attention-replay-20261020/main.rs b/.agents/runs/model-execution-original-attention-replay-20261020/main.rs
new file mode 100644
--- /dev/null
+++ b/.agents/runs/model-execution-original-attention-replay-20261020/main.rs
@@ -0,0 +1,634 @@
+//! ROOT ONLY: authentic private attention operators; no token/session engine.
+use serde_json::{Value, json};
+use sha2::{Digest, Sha256};
+use std::collections::BTreeMap;
+use std::error::Error;
+use std::fs::OpenOptions;
+use std::io::{Read, Write};
+use std::os::unix::fs::OpenOptionsExt;
+use std::path::{Component, Path};
+use std::sync::Arc;
+use std::sync::atomic::{AtomicBool, Ordering};
+use std::thread;
+use zerogravity_artifacts::artifact::Artifact;
+use zerogravity_artifacts::cache::TransferAdmission;
+use zerogravity_artifacts::source_weights::{
+    ExpectedSourceShard, SourceWeights, VerifiedTensorSource, WindowOpenLimits,
+};
+use zerogravity_core::accelerator::program::{ActivationPrecision, ProgramNumericalProfile};
+use zerogravity_core::resource_arbiter::{
+    AllocationClass, AllocationRequest, DeviceBudgetSnapshot, PhysicalDeviceKey, ResourceArbiter,
+    ResourceOwner,
+};
+use zerogravity_model_qwen35::operator_slice::attention::{
+    AttentionProjectionNormOutput, AttentionSliceInput,
+};
+use zerogravity_model_qwen35::original_binding::{OriginalWindowOperators, QwenNumericalProfile};
+
+type Proof<T> = Result<T, Box<dyn Error>>;
+const MIB: u64 = 1024 * 1024;
+const HOST: u64 = 128 * MIB;
+const SOURCE_META: u64 = 64 * MIB;
+const CONTROL: u64 = 2 * MIB;
+const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
+const STAGES: [&str; 7] = [
+    "query_gate_projection",
+    "key_projection",
+    "value_projection",
+    "query",
+    "gate",
+    "normalized_query",
+    "normalized_key",
+];
+
+fn ensure(ok: bool, message: impl Into<String>) -> Proof<()> {
+    if ok {
+        Ok(())
+    } else {
+        Err(message.into().into())
+    }
+}
+fn text(value: &Value) -> Proof<&str> {
+    value
+        .as_str()
+        .ok_or_else(|| "required string missing".into())
+}
+fn sha(bytes: &[u8]) -> String {
+    format!("{:x}", Sha256::digest(bytes))
+}
+fn read(path: &Path, cap: usize) -> Proof<Vec<u8>> {
+    let file = OpenOptions::new()
+        .read(true)
+        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+        .open(path)?;
+    ensure(
+        file.metadata()?.is_file() && file.metadata()?.len() <= cap as u64,
+        "bounded regular input",
+    )?;
+    let mut raw = Vec::new();
+    file.take(cap as u64 + 1).read_to_end(&mut raw)?;
+    ensure(raw.len() <= cap, "bounded input grew")?;
+    Ok(raw)
+}
+fn pinned(path: &Path, pin: &Value, cap: usize) -> Proof<Vec<u8>> {
+    let raw = read(path, cap)?;
+    ensure(
+        pin["bytes"].as_u64() == Some(raw.len() as u64) && pin["sha256"] == sha(&raw),
+        "exact input file digest/length",
+    )?;
+    Ok(raw)
+}
+fn relative(value: &Value) -> Proof<&Path> {
+    let path = Path::new(text(value)?);
+    ensure(
+        !path.as_os_str().is_empty()
+            && path
+                .components()
+                .all(|part| matches!(part, Component::Normal(_))),
+        "contained input name",
+    )?;
+    Ok(path)
+}
+fn write_json(path: &Path, value: &Value) -> Proof<()> {
+    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
+    serde_json::to_writer_pretty(&mut output, value)?;
+    output.write_all(b"\n")?;
+    Ok(())
+}
+fn snapshot(host: &TransferAdmission) -> Proof<DeviceBudgetSnapshot> {
+    Ok(host.arbiter.snapshot(&host.resource)?)
+}
+fn class(row: &DeviceBudgetSnapshot, kind: AllocationClass) -> u64 {
+    row.per_class.get(&kind).copied().unwrap_or(0)
+}
+fn ledger(row: &DeviceBudgetSnapshot) -> Value {
+    json!({"reserved":row.reserved,"remaining":row.remaining,"reservations":row.reservations,
+        "fenced":row.fenced,"per_class":row.per_class.iter()
+            .map(|(k,v)|(format!("{k:?}"),*v)).collect::<BTreeMap<_,_>>()})
+}
+fn same_ledger(host: &TransferAdmission, before: &DeviceBudgetSnapshot) -> Proof<()> {
+    let after = snapshot(host)?;
+    ensure(
+        after.reserved == before.reserved
+            && after.per_class == before.per_class
+            && after.reservations == before.reservations
+            && !after.fenced,
+        "all private failed-call buffers released",
+    )
+}
+fn source(binding: &Value, host: &TransferAdmission) -> Proof<(VerifiedTensorSource, Vec<u8>)> {
+    let root = Path::new(text(&binding["source_directory"])?);
+    let files = binding["source_files"]
+        .as_array()
+        .ok_or("original source file pins")?;
+    let get = |name: &str| -> Proof<&Value> {
+        files
+            .iter()
+            .find(|row| row["path"] == name)
+            .ok_or_else(|| "required source file pin".into())
+    };
+    let config = pinned(&root.join("config.json"), get("config.json")?, MIB as usize)?;
+    let index = pinned(
+        &root.join("model.safetensors.index.json"),
+        get("model.safetensors.index.json")?,
+        MIB as usize,
+    )?;
+    let parsed: Value = serde_json::from_slice(&index)?;
+    let shard_names = parsed["weight_map"]
+        .as_object()
+        .ok_or("complete original index")?
+        .values()
+        .map(text)
+        .collect::<Proof<std::collections::BTreeSet<_>>>()?;
+    ensure(shard_names.len() == 1, "approved single source shard")?;
+    let mut shards = Vec::new();
+    for name in shard_names {
+        relative(&json!(name))?;
+        let expected = get(name)?;
+        let file = OpenOptions::new()
+            .read(true)
+            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
+            .open(root.join(name))?;
+        ensure(
+            file.metadata()?.is_file()
+                && file.metadata()?.len() == expected["bytes"].as_u64().ok_or("shard length")?,
+            "same original full regular shard",
+        )?;
+        shards.push(ExpectedSourceShard::new(
+            name.into(),
+            Artifact::from_file(file, Path::new(name))?,
+            expected["bytes"].as_u64().ok_or("shard length")?,
+            text(&expected["sha256"])?.into(),
+        )?);
+    }
+    let source = SourceWeights::prepare_verified_windows(
+        &index,
+        text(&get("model.safetensors.index.json")?["sha256"])?,
+        shards,
+        WindowOpenLimits {
+            max_source_bytes: 5 * 1024 * MIB,
+            max_metadata_bytes: SOURCE_META,
+            max_shards: 1,
+            max_tensors: 1024,
+            scan_chunk_bytes: MIB as usize,
+        },
+        host,
+    )?
+    .open()?;
+    ensure(
+        source.metadata_reserved_bytes() == SOURCE_META && source.source_identity().is_bound(),
+        "sealed original issuer and admitted metadata",
+    )?;
+    Ok((source, config))
+}
+fn input<'a>(source: &VerifiedTensorSource, raw: &'a [u8]) -> Proof<AttentionSliceInput<'a>> {
+    Ok(AttentionSliceInput::from_bf16_bytes(
+        source.source_identity(),
+        PROFILE,
+        raw,
+    )?)
+}
+fn refusal<T>(result: zerogravity_core::Result<T>, expected: &str) -> Proof<()> {
+    match result {
+        Err(error) if error.to_string().contains(expected) => Ok(()),
+        Err(error) => Err(format!("wrong refusal: {error}").into()),
+        Ok(value) => {
+            drop(value);
+            Err("refused call returned private output".into())
+        }
+    }
+}
+
+// The actual admitted stage, never a sleep or elapsed-token guess, arms cancel.
+struct CancelMonitor {
+    stop: Arc<AtomicBool>,
+    flag: Arc<AtomicBool>,
+    join: Option<thread::JoinHandle<Value>>,
+}
+impl CancelMonitor {
+    fn start(host: TransferAdmission) -> Self {
+        let stop = Arc::new(AtomicBool::new(false));
+        let flag = Arc::new(AtomicBool::new(false));
+        let (stopped, cancelled) = (stop.clone(), flag.clone());
+        let join = thread::spawn(move || {
+            while !stopped.load(Ordering::Acquire) {
+                let row = match host.arbiter.snapshot(&host.resource) {
+                    Ok(row) => row,
+                    Err(error) => return json!({"error":error.to_string()}),
+                };
+                if class(&row, AllocationClass::Output) >= 2 * 64 * 1024
+                    && class(&row, AllocationClass::TransferStaging) > 0
+                {
+                    cancelled.store(true, Ordering::Release);
+                    return ledger(&row);
+                }
+                thread::yield_now();
+            }
+            Value::Null
+        });
+        Self {
+            stop,
+            flag,
+            join: Some(join),
+        }
+    }
+    fn finish(mut self) -> Proof<Value> {
+        self.stop.store(true, Ordering::Release);
+        self.join
+            .take()
+            .ok_or("cancel observer missing")?
+            .join()
+            .map_err(|_| "cancel observer panic".into())
+    }
+}
+impl Drop for CancelMonitor {
+    fn drop(&mut self) {
+        self.stop.store(true, Ordering::Release);
+        if let Some(join) = self.join.take() {
+            let _ = join.join();
+        }
+    }
+}
+fn guards(
+    operators: &OriginalWindowOperators,
+    source: &VerifiedTensorSource,
+    host: &TransferAdmission,
+    bytes: &[u8],
+) -> Proof<Value> {
+    let before = snapshot(host)?;
+    refusal(
+        operators.execute_attention_projection_norm_slice(
+            3,
+            input(source, bytes)?,
+            Some(&AtomicBool::new(true)),
+        ),
+        "cancelled",
+    )?;
+    same_ledger(host, &before)?;
+    ensure(
+        operators
+            .execute_attention_projection_norm_slice(0, input(source, bytes)?, None)
+            .is_err(),
+        "GDN layer cannot use attention slice",
+    )?;
+    same_ledger(host, &before)?;
+    let mut held = Vec::new();
+    loop {
+        let left = snapshot(host)?.remaining;
+        if left == 0 {
+            break;
+        }
+        held.push(source.reserve_workspace(left.min(32 * MIB) as usize)?);
+    }
+    ensure(snapshot(host)?.reserved == HOST, "real pool exhaustion")?;
+    refusal(
+        operators.execute_attention_projection_norm_slice(3, input(source, bytes)?, None),
+        "limit",
+    )?;
+    ensure(
+        snapshot(host)?.reserved == HOST,
+        "no leaked pressure candidate",
+    )?;
+    drop(held);
+    same_ledger(host, &before)?;
+    let monitor = CancelMonitor::start(host.clone());
+    let cancelled = operators.execute_attention_projection_norm_slice(
+        3,
+        input(source, bytes)?,
+        Some(&monitor.flag),
+    );
+    let evidence = monitor.finish()?;
+    ensure(
+        evidence.is_object() && evidence.get("error").is_none(),
+        "actual projection-stage cancellation observed",
+    )?;
+    refusal(cancelled, "cancelled")?;
+    same_ledger(host, &before)?;
+    Ok(
+        json!({"pre_cancel":true,"wrong_layer":true,"physical_pool_pressure":true,
+        "mid_cancel_actual_projection_stage":evidence,"failed_call_reservations_restored":true}),
+    )
+}
+fn values(output: &AttentionProjectionNormOutput) -> Proof<[&[f32]; 7]> {
+    Ok([
+        output.query_gate_projection()?,
+        output.key_projection()?,
+        output.value_projection()?,
+        output.query()?,
+        output.gate()?,
+        output.normalized_query()?,
+        output.normalized_key()?,
+    ])
+}
+fn raw_reference(root: &Path, row: &Value) -> Proof<Vec<u8>> {
+    ensure(row["dtype"] == "BF16", "raw BF16 reference")?;
+    let raw = pinned(&root.join(relative(&row["file"])?), row, 32768)?;
+    ensure(
+        !raw.is_empty()
+            && raw.len().is_multiple_of(2)
+            && raw
+                .chunks_exact(2)
+                .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80),
+        "finite exact BF16 reference",
+    )?;
+    Ok(raw)
+}
+fn compare(actual: &[f32], expected: &[u8]) -> Proof<Value> {
+    ensure(actual.len() * 2 == expected.len(), "full raw output length")?;
+    let mut different = 0;
+    let mut first = Value::Null;
+    for (index, (value, bytes)) in actual.iter().zip(expected.chunks_exact(2)).enumerate() {
+        let bits = value.to_bits();
+        let reference = u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16;
+        if bits != reference {
+            different += 1;
+            if first.is_null() {
+                first = json!({"index":index,"actual_f32_bits":format!("{bits:08x}"),
+                "expected_bf16_widened_bits":format!("{reference:08x}")});
+            }
+        }
+        ensure(
+            value.is_finite() && bits & 0xffff == 0,
+            "actual finite BF16-widened output",
+        )?;
+    }
+    Ok(
+        json!({"words":actual.len(),"different_words":different,"first_difference":first,"tolerance":null}),
+    )
+}
+fn check_case(
+    output: &AttentionProjectionNormOutput,
+    case: &Value,
+    root: &Path,
+    output_dir: &Path,
+    model: &str,
+    after_drop: bool,
+) -> Proof<Value> {
+    ensure(
+        output.layer() == 3
+            && output.precision() == ActivationPrecision::Bf16WidenedF32
+            && output.numerical_profile() == PROFILE,
+        "original typed fragment result",
+    )?;
+    let expected_input = raw_reference(root, &case["tensors"]["hidden"])?;
+    ensure(
+        compare(output.input()?, &expected_input)?["different_words"] == 0,
+        "authentic input bytes preserved",
+    )?;
+    let mut records = Vec::new();
+    let mut mismatches = 0;
+    for (stage, actual) in STAGES.into_iter().zip(values(output)?) {
+        let expected = raw_reference(root, &case["tensors"][stage])?;
+        let compared = compare(actual, &expected)?;
+        mismatches += compared["different_words"]
+            .as_u64()
+            .ok_or("difference count")?;
+        let name = format!(
+            "{model}-step-{}-{stage}.bf16",
+            case["step"].as_u64().ok_or("step")?
+        );
+        // Pure byte transport of already BF16-widened original HostMath output.
+        let raw = actual
+            .iter()
+            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
+            .collect::<Vec<_>>();
+        if !after_drop {
+            OpenOptions::new()
+                .write(true)
+                .create_new(true)
+                .open(output_dir.join(&name))?
+                .write_all(&raw)?;
+        }
+        records.push(json!({"stage":stage,"file":name,"bytes":raw.len(),"sha256":sha(&raw),
+            "comparison":compared,"source_tensor_label":case["tensors"][stage]["source_tensor_label"]}));
+    }
+    let record = json!({"model":model,"step":case["step"],"teacher_token":case["teacher_token"],
+        "layer":3,"profile":"source-36-f32-restored","after_callers_drop":after_drop,
+        "different_words":mismatches,"stages":records});
+    write_json(
+        &output_dir.join(format!(
+            "{model}-step-{}-comparison-{}.json",
+            case["step"], after_drop
+        )),
+        &record,
+    )?;
+    ensure(
+        mismatches == 0,
+        "full seven-output raw BF16 comparison failed; see retained first differences",
+    )?;
+    Ok(record)
+}
+fn exercise(
+    model: &str,
+    binding: &Value,
+    references: &Value,
+    inputs_root: &Path,
+    output: &Path,
+) -> Proof<Value> {
+    let arbiter = ResourceArbiter::new();
+    let key = PhysicalDeviceKey::new("host", "original-attention-replay")?;
+    arbiter.register(key.clone(), HOST)?;
+    let host = TransferAdmission {
+        arbiter,
+        resource: key,
+        owner: ResourceOwner::new(model, "attention-operators")?,
+    };
+    let control = host.arbiter.admit(
+        &host.resource,
+        host.owner.clone(),
+        &[AllocationRequest {
+            kind: AllocationClass::Other,
+            bytes: CONTROL,
+        }],
+    )?;
+    let (source, config) = source(binding, &host)?;
+    let identity = source.source_identity();
+    let files = binding["source_files"].as_array().ok_or("source files")?;
+    let digest = |name: &str| -> Proof<&str> {
+        text(
+            &files
+                .iter()
+                .find(|v| v["path"] == name)
+                .ok_or("source pin")?["sha256"],
+        )
+    };
+    let operators = OriginalWindowOperators::bind(
+        source.clone(),
+        &config,
+        digest("config.json")?,
+        digest("model.safetensors.index.json")?,
+        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
+    )?;
+    let geometry = operators.binding().config().geometry();
+    ensure(
+        operators.binding().text_tensor_count() == 320
+            && geometry.hidden_size == binding["hidden"].as_u64().ok_or("hidden")?
+            && geometry.num_attention_heads == 8
+            && geometry.num_key_value_heads == 2
+            && geometry.head_dim == 256,
+        "actual source attention geometry",
+    )?;
+    let before = snapshot(&host)?;
+    ensure(
+        before.reserved == SOURCE_META + 4 * MIB + CONTROL
+            && class(&before, AllocationClass::Output) == 0,
+        "source/binding/control steady exact reservation",
+    )?;
+    let cases = references["cases"]
+        .as_array()
+        .ok_or("two authentic cases")?;
+    ensure(cases.len() == 2, "both teachers present")?;
+    let bytes = raw_reference(inputs_root, &cases[0]["tensors"]["hidden"])?;
+    let guard_report = guards(&operators, &source, &host, &bytes)?;
+    let mut results = Vec::new();
+    let mut outputs = Vec::new();
+    for (step, case) in cases.iter().enumerate() {
+        ensure(
+            case["step"] == step
+                && case["teacher_token"] == [11, 353][step]
+                && case["layer"] == 3
+                && case["profile"] == "source-36-f32-restored",
+            "exact teacher provenance",
+        )?;
+        let bytes = raw_reference(inputs_root, &case["tensors"]["hidden"])?;
+        let result =
+            operators.execute_attention_projection_norm_slice(3, input(&source, &bytes)?, None)?;
+        ensure(
+            result.source_identity() == identity,
+            "independent original issuer preserved",
+        )?;
+        results.push(check_case(
+            &result,
+            case,
+            inputs_root,
+            output,
+            model,
+            false,
+        )?);
+        outputs.push(result);
+        let live = snapshot(&host)?;
+        ensure(
+            live.reserved == before.reserved + outputs.len() as u64 * 8 * MIB
+                && class(&live, AllocationClass::Output) == outputs.len() as u64 * 8 * 64 * 1024
+                && class(&live, AllocationClass::TransferStaging) == 0,
+            "exact retained eight private output backings per case",
+        )?;
+    }
+    drop(operators);
+    drop(source);
+    drop(config);
+    let retained = snapshot(&host)?;
+    ensure(
+        retained.reserved == SOURCE_META + CONTROL + 16 * MIB
+            && class(&retained, AllocationClass::Output) == 16 * 64 * 1024,
+        "caller dropped while original issuer/output leases retained",
+    )?;
+    for (result, case) in outputs.iter().zip(cases) {
+        check_case(result, case, inputs_root, output, model, true)?;
+    }
+    drop(outputs);
+    ensure(
+        snapshot(&host)?.reserved == CONTROL,
+        "all source/output backing released after result drop",
+    )?;
+    drop(control);
+    let final_row = snapshot(&host)?;
+    ensure(
+        final_row.reserved == 0 && final_row.reservations == 0 && !final_row.fenced,
+        "actual final arbiter zero",
+    )?;
+    let compared_words = results
+        .iter()
+        .map(|case| {
+            case["stages"]
+                .as_array()
+                .ok_or("stage records")?
+                .iter()
+                .map(|stage| {
+                    stage["comparison"]["words"]
+                        .as_u64()
+                        .ok_or("actual stage word count")
+                })
+                .try_fold(0u64, |sum, words| Ok::<_, &str>(sum + words?))
+        })
+        .try_fold(0u64, |sum, words| Ok::<_, &str>(sum + words?))?;
+    ensure(
+        compared_words == 23552,
+        "complete actual seven-output word count",
+    )?;
+    Ok(
+        json!({"model":model,"source_identity":format!("{identity:?}"),"guards":guard_report,
+        "cases":results,"retained_after_callers_drop":ledger(&retained),"final_ledger":ledger(&final_row),
+        "compared_output_words":compared_words,"source_window_payloads_immutable":true}),
+    )
+}
+fn run() -> Proof<Value> {
+    ensure(
+        !cfg!(debug_assertions),
+        "ROOT physical operator replay requires release",
+    )?;
+    let args = std::env::args().skip(1).collect::<Vec<_>>();
+    ensure(
+        args.len() == 4,
+        "inputs.json inputsSHA256 output-directory model(0.8b|2b)",
+    )?;
+    let inputs = Path::new(&args[0]);
+    let raw = read(inputs, MIB as usize)?;
+    ensure(
+        sha(&raw) == args[1],
+        "actual authenticated exported manifest pin",
+    )?;
+    let references: Value = serde_json::from_slice(&raw)?;
+    let binding: Value = serde_json::from_str(include_str!("bindings.json"))?;
+    ensure(
+        references["schema"] == "zerogravity.original-attention-fragment.raw-inputs.v1"
+            && references["bindings_sha256"] == sha(include_bytes!("bindings.json"))
+            && references["profile"] == "source-36-f32-restored"
+            && references["layer"] == 3
+            && references["teacher_tokens"] == json!([11, 353])
+            && references["payload_bytes"] == 106496,
+        "frozen authentic external input binding",
+    )?;
+    ensure(
+        matches!(args[3].as_str(), "0.8b" | "2b"),
+        "approved original source",
+    )?;
+    let actual_affinity = std::fs::read_to_string("/proc/self/status")?
+        .lines()
+        .find(|line| line.starts_with("Cpus_allowed_list:"))
+        .ok_or("actual affinity unavailable")?
+        .to_owned();
+    let model = &args[3];
+    let result = exercise(
+        model,
+        &binding["models"][model],
+        &references["models"][model],
+        inputs.parent().ok_or("reference root")?,
+        Path::new(&args[2]),
+    )?;
+    Ok(
+        json!({"status":"PASS_PRIVATE_ATTENTION_FRAGMENT_ONLY","result":result,
+        "inputs_sha256":args[1],"actual_affinity":actual_affinity,
+        "full_HF_or_StateRoot_or_GPU_or_performance_qualified":false}),
+    )
+}
+fn main() {
+    let output = std::env::args().nth(3);
+    let result = run();
+    let report = match &result {
+        Ok(report) => report.clone(),
+        Err(error) => json!({"status":"FAIL_UNQUALIFIED","error":error.to_string(),
+            "full_HF_or_StateRoot_or_GPU_or_performance_qualified":false}),
+    };
+    let written = output
+        .as_deref()
+        .ok_or("output path missing")
+        .and_then(|path| {
+            write_json(&Path::new(path).join("report.json"), &report)
+                .map_err(|_| "report write failed")
+        });
+    println!("{report}");
+    if result.is_err() || written.is_err() {
+        std::process::exit(1);
+    }
+}
diff --git a/.agents/runs/model-execution-original-attention-replay-20261020/owned_lifetime.py b/.agents/runs/model-execution-original-attention-replay-20261020/owned_lifetime.py
new file mode 100644
--- /dev/null
+++ b/.agents/runs/model-execution-original-attention-replay-20261020/owned_lifetime.py
@@ -0,0 +1,91 @@
+"""Exact accepted owned-native leader cleanup functions; no child on import."""
+import os
+import resource
+import signal
+import time
+from contracts import require
+CAP = 4 * 1024 * 1024
+
+
+def unreaped(child):
+    require(child.returncode is None, "leader already reaped")
+    value = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+    if value is None:
+        return None
+    require(value.si_pid == child.pid, "owned child identity")
+    return value.si_status if value.si_code == os.CLD_EXITED else -value.si_status
+
+
+def cleanup_owned(child, record, errors, *, observe=unreaped, pause=time.sleep):
+    """Independent bounded stages; never signal after the first reap attempt."""
+    def bound():
+        observe(child)
+        require(os.getpgid(child.pid) == child.pid and os.getsid(child.pid) == child.pid,
+                "owned session/group binding")
+        return child.pid
+
+    try:
+        os.killpg(bound(), signal.SIGTERM)
+        record["group_term_before_reap"] = True
+    except ProcessLookupError:
+        record["group_term_absent"] = True
+    except BaseException as error:
+        errors.append("owned TERM: " + repr(error))
+    try:
+        grace = time.monotonic() + 0.5
+        while time.monotonic() < grace:
+            observe(child)
+            pause(0.05)
+    except BaseException as error:
+        errors.append("owned grace: " + repr(error))
+    # TERM/grace/one KILL error cannot skip the separately bound KILL stage.
+    # Both attempts occur before ANY wait, while the owned leader is unreaped.
+    for attempt in range(2):
+        try:
+            os.killpg(bound(), signal.SIGKILL)
+            record["group_kill_before_reap"] = True
+            break
+        except ProcessLookupError:
+            record["group_kill_absent_before_reap"] = True
+            break
+        except BaseException as error:
+            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
+    # ALWAYS attempt reap even after signal/ownership failures. Popen.wait only
+    # targets this child. An interrupted first wait cannot skip the final wait.
+    # No signal operation occurs below this point, including after timeout.
+    try:
+        record["reaped_exit_code"] = child.wait(timeout=5)
+        record["leader_reaped"] = True
+    except BaseException as error:
+        errors.append("owned initial reap: " + repr(error))
+    finally:
+        if child.returncode is None:
+            try:
+                record["reaped_exit_code"] = child.wait(timeout=5)
+                record["leader_reaped"] = True
+            except BaseException as error:
+                errors.append("owned final reap: " + repr(error))
+        else:
+            record["reaped_exit_code"] = child.returncode
+            record["leader_reaped"] = True
+    if record.get("leader_reaped"):
+        try:
+            deadline = time.monotonic() + 5
+            while True:
+                try:
+                    os.killpg(child.pid, 0)  # Read-only absence check, NEVER kill after reap.
+                except ProcessLookupError:
+                    record["group_absent"] = True
+                    break
+                require(time.monotonic() < deadline, "owned process group remains")
+                pause(0.05)
+        except BaseException as error:
+            errors.append("owned group absence: " + repr(error))
+
+
+def launch_limits():
+    # Python is single-threaded at Popen; no foreign runtime is imported.
+    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
+    resource.setrlimit(resource.RLIMIT_CPU, (3601, 3601))
+    resource.setrlimit(resource.RLIMIT_FSIZE, (CAP, CAP))
+    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
diff --git a/.agents/runs/model-execution-original-attention-replay-20261020/prepare_inputs.py b/.agents/runs/model-execution-original-attention-replay-20261020/prepare_inputs.py
new file mode 100644
--- /dev/null
+++ b/.agents/runs/model-execution-original-attention-replay-20261020/prepare_inputs.py
@@ -0,0 +1,120 @@
+"""ROOT ONLY: export authenticated existing CPU words, no Torch/operator/model execution."""
+import argparse
+import hashlib
+from pathlib import Path
+
+from contracts import (HERE, LABELS_8, MODELING, PROFILE, SHAPES, STAGES, TOKENS, archive,
+                       payload, pin, read_json, require, sha, source_2, source_8, write)
+
+
+def metadata(row):
+    pin(row["path"], row)
+    return read_json(row["path"])
+
+
+def lifecycle(invocation, wide):
+    require(invocation["exit_code"] == 0, "original observer exited successfully")
+    if wide:
+        cleanup = invocation["foreground_cleanup"]
+        require(invocation["cleanup_errors"] == [] and invocation["owned_container_absent_after_reap"] is True
+                and cleanup["stage_errors"] == [] and cleanup["leader_reaped"] is True
+                and cleanup["group_absent"] is True and cleanup["all_nonzero_signals_before_reap"] is True
+                and invocation["log_cleanup"]["pipe_eof"] is True
+                and invocation["log_cleanup"]["all_closed"] is True, "accepted 2B owned lifecycle")
+    else:
+        require(invocation["source_public_oracle_passed"] is True
+                and invocation["cleanup"]["failures"] == []
+                and invocation["cleanup"]["container_absence_proven"] is True
+                and invocation["cleanup"]["owned_process_reaped"] is True, "accepted .8B observer lifecycle")
+
+
+def parents(bindings):
+    require(bindings["schema"] == "zerogravity.original-attention-fragment.bindings.v1"
+            and bindings["profile"] == PROFILE and bindings["layer"] == 3
+            and bindings["teacher_tokens"] == list(TOKENS), "frozen exact scope")
+    for source in bindings["protocol_sources"]:
+        pin(source["path"], source)
+    # Authenticate every metadata parent before selecting any source tensor.
+    receipt8 = metadata(bindings["models"]["0.8b"]["report"])
+    lifecycle(metadata(bindings["models"]["0.8b"]["invocation"]), False)
+    model2 = bindings["models"]["2b"]
+    paired, manifest = metadata(model2["paired"]), metadata(model2["manifest"])
+    metadata(model2["source_pins"])
+    accepted = metadata(model2["accepted_validation"])
+    require(accepted["archive_validation"] == "PASS" and accepted["counts"] == {
+        "canonical_domains": 168, "full_vocab_logits": 4, "observed_intermediate_tensors": 60,
+        "profiles": 2, "step_archives": 4}, "complete unchanged accepted reference validation")
+    require(model2["manifest"]["sha256"] == paired["counts"]["observed"]["manifest_sha256"],
+            "paired observed manifest closure")
+    lifecycle(metadata(model2["plain_invocation"]), True)
+    lifecycle(metadata(model2["observed_invocation"]), True)
+    return receipt8, paired, manifest
+
+
+def prepare(output):
+    bindings = read_json(HERE / "bindings.json")
+    receipt8, paired, manifest = parents(bindings)
+    output = Path(output)
+    require(output.resolve().parent == HERE.resolve(), "fresh extract directly under this run")
+    output.mkdir(mode=0o700)
+    result = {"schema": "zerogravity.original-attention-fragment.raw-inputs.v1",
+              "status": "AUTHENTIC_CPU_OPERATOR_INPUTS_ONLY_RUST_UNQUALIFIED",
+              "bindings_sha256": sha(HERE / "bindings.json"), "profile": PROFILE, "layer": 3,
+              "teacher_tokens": list(TOKENS), "models": {}, "payload_bytes": 0,
+              "math_reimplemented": False, "new_model_or_operator_execution": False}
+    for model, binding in bindings["models"].items():
+        cases = []
+        for step, row in enumerate(binding["steps"]):
+            meta = metadata(row["metadata"])
+            if model == "0.8b":
+                source_8(receipt8, step, meta)
+                require(row["archive"]["sha256"] == meta["archive_sha256"], "actual .8B archive closure")
+            else:
+                source_2(paired, manifest, step, meta)
+                require(row["archive"]["sha256"] == meta["archive"]["sha256"], "actual 2B archive closure")
+            path = pin(row["archive"]["path"], row["archive"])
+            tensors = archive(path)
+            case = {"step": step, "teacher_token": TOKENS[step], "profile": PROFILE, "layer": 3,
+                    "archive": row["archive"], "metadata": row["metadata"], "tensors": {}}
+            for stage in ("hidden",) + STAGES:
+                label = LABELS_8[stage] if model == "0.8b" else "layer-03." + stage
+                shape = [1, 1, binding["hidden"]] if stage == "hidden" else SHAPES[stage]
+                actual, observer = tensors[label], meta["tensors"][label]
+                require(observer["dtype"] == "torch.bfloat16" and observer["shape"] == shape
+                        and observer["bytes"] == actual["bytes"], "actual observer dtype/shape/full length")
+                if model == "0.8b":
+                    if stage == "query":
+                        require(observer["stride"][-2:] == [512, 1], "actual interleaved query source view")
+                else:
+                    require(observer["source_module_layer"] == 3 and observer["stage"] == stage,
+                            "actual public module layer/stage")
+                    if stage == "query":
+                        require(observer["original_stride"][-2:] == [512, 1], "actual interleaved query source view")
+                raw = payload(path, actual, shape)
+                digest = hashlib.sha256(raw).hexdigest()
+                if model == "0.8b":
+                    require(digest == observer["sha256"], "actual untouched tensor observer digest")
+                name = f"{model}-step-{step}-{stage}.bf16"
+                with (output / name).open("xb") as opened:
+                    opened.write(raw)
+                case["tensors"][stage] = {"file": name, "bytes": len(raw), "sha256": digest,
+                    "dtype": "BF16", "shape": shape, "source_tensor_label": label,
+                    "archive_offset": actual["offset"], "source_observer": observer}
+                result["payload_bytes"] += len(raw)
+            # Authentic normalized K and original V/key heads have their exact
+            # logical byte order; no expected tensor is reconstructed from QG.
+            pin(path, row["archive"])
+            pin(row["metadata"]["path"], row["metadata"])
+            cases.append(case)
+        result["models"][model] = {"source_repo": binding["repo"], "source_revision": binding["revision"],
+                                  "hidden": binding["hidden"], "cases": cases}
+    require(result["payload_bytes"] == 106496, "complete nonvacuous two-model/two-teacher inputs plus seven outputs")
+    write(output / "inputs.json", result)
+    print({"status": result["status"], "bytes": result["payload_bytes"],
+           "inputs_sha256": sha(output / "inputs.json")})
+
+
+if __name__ == "__main__":
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument("--new-output", required=True, type=Path)
+    prepare(parser.parse_args().new_output)
diff --git a/.agents/runs/model-execution-original-attention-replay-20261020/run_replay.py b/.agents/runs/model-execution-original-attention-replay-20261020/run_replay.py
new file mode 100644
--- /dev/null
+++ b/.agents/runs/model-execution-original-attention-replay-20261020/run_replay.py
@@ -0,0 +1,181 @@
+"""ROOT ONLY native CPU helper orchestration. No Torch/model/container imports."""
+import argparse
+import json
+import os
+from pathlib import Path
+import selectors
+import signal
+import stat
+import subprocess
+import time
+
+from contracts import HERE, pin, read_json, require, sha, write
+from owned_lifetime import cleanup_owned, launch_limits, unreaped
+
+CAP = 4 * 1024 * 1024
+
+
+def source_files(binding):
+    """ROOT performs complete source scans; never provider/opaque-memory math."""
+    root = Path(binding["source_directory"])
+    rows = []
+    for item in binding["source_files"]:
+        path = root / item["path"]
+        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
+        with os.fdopen(os.open(path, flags), "rb") as opened:
+            before = os.fstat(opened.fileno())
+            require(stat.S_ISREG(before.st_mode) and before.st_size == item["bytes"], "original source regular FD/full length")
+            import hashlib
+            digest = hashlib.file_digest(opened, "sha256").hexdigest()
+            after = os.fstat(opened.fileno())
+        identity = lambda value: [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
+        require(identity(before) == identity(after) and digest == item["sha256"], "complete original source FD digest/identity")
+        rows.append({"path": str(path), "bytes": before.st_size, "sha256": digest,
+                     "fd_identity": identity(before)})
+    return rows
+
+
+def frozen():
+    manifest = read_json(HERE / "freeze-manifest.json")
+    for item in manifest["files"]:
+        pin(HERE / item["file"], item)
+    for item in manifest["main_sources"]:
+        pin(item["path"], item)
+    return sha(HERE / "freeze-manifest.json")
+
+
+def execute(binary, binary_fd, inputs, inputs_sha, output, model, cpus, freeze, source_before):
+    argv = [str(binary), str(inputs), inputs_sha, str(output), model]
+    environment = {key: value for key, value in os.environ.items()
+                   if key in ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "XDG_CACHE_HOME")}
+    environment.update(RAYON_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
+                       CUDA_VISIBLE_DEVICES="-1", NVIDIA_VISIBLE_DEVICES="void",
+                       ATEN_CPU_CAPABILITY="avx2")
+    record = {"status": "UNQUALIFIED", "argv": argv, "freeze_manifest_sha256": freeze,
+              "binary_sha256": sha(binary), "cpu_affinity": cpus, "scope": "seven private attention operators only",
+              "address_space_cap_bytes": 512 * 1024**2, "wall_cap_seconds": 900,
+              "log_cap_bytes": CAP, "source_before": source_before,
+              "no_model_or_session_or_GPU_execution": True}
+    child = selector = pipe = log = None
+    errors, primary, eof, retained = [], None, False, 0
+    started = time.monotonic()
+    try:
+        log = (output / "execution.log").open("xb")
+        child = subprocess.Popen(argv, executable=f"/proc/self/fd/{binary_fd}", pass_fds=(binary_fd,),
+            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
+            env=environment, start_new_session=True, preexec_fn=launch_limits)
+        # Every post-spawn operation is covered by cleanup below.
+        record["owned_cli_pid"] = child.pid
+        pipe = child.stdout
+        os.set_blocking(pipe.fileno(), False)
+        selector = selectors.DefaultSelector()
+        selector.register(pipe, selectors.EVENT_READ)
+
+        def drain(timeout):
+            nonlocal retained, eof
+            for key, _ in selector.select(timeout):
+                try:
+                    data = os.read(key.fd, 65536)
+                except BlockingIOError:
+                    continue
+                if not data:
+                    selector.unregister(key.fileobj)
+                    eof = True
+                else:
+                    require(retained + len(data) <= CAP, "bounded child combined log")
+                    log.write(data)
+                    retained += len(data)
+        while True:
+            drain(0.05)
+            status = unreaped(child)
+            if status is not None:
+                record["exit_code"] = status
+                break
+            require(time.monotonic() - started < 900, "finite helper deadline")
+    except BaseException as error:
+        primary = repr(error)
+    finally:
+        if child is not None:
+            cleanup_owned(child, record, errors)
+        if selector is not None:
+            try:
+                deadline = time.monotonic() + 2
+                while not eof and time.monotonic() < deadline:
+                    drain(0.05)
+                require(eof, "owned pipe EOF deadline")
+            except BaseException as error:
+                errors.append("final drain: " + repr(error))
+        for name, owned in (("selector", selector), ("pipe", pipe), ("log", log)):
+            if owned is not None:
+                try:
+                    owned.close()
+                    record[name + "_closed"] = True
+                except BaseException as error:
+                    errors.append(name + " close: " + repr(error))
+        record.update(primary_error=primary, cleanup_errors=errors, pipe_eof=eof, log_bytes=retained,
+                      elapsed_seconds_not_benchmark=time.monotonic() - started)
+    success = primary is None and not errors and record.get("exit_code") == record.get("reaped_exit_code") == 0 \
+        and record.get("leader_reaped") and record.get("group_absent") and eof
+    record["status"] = "CHILD_EXECUTED_REQUIRES_VALIDATION" if success else "FAIL_UNQUALIFIED"
+    write(output / "invocation.json", record)
+    require(success, "owned native helper failed; preserve fresh attempt")
+    return record
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument("--binary", type=Path, required=True)
+    parser.add_argument("--binary-sha256", required=True)
+    parser.add_argument("--inputs", type=Path, required=True)
+    parser.add_argument("--inputs-sha256", required=True)
+    parser.add_argument("--new-output", type=Path, required=True)
+    args = parser.parse_args()
+    def deadline(_signal, _frame):
+        raise TimeoutError("finite source-audit and owned-helper overall deadline")
+    signal.signal(signal.SIGALRM, deadline)
+    signal.alarm(3300)
+    freeze = frozen()
+    require(args.inputs.resolve().parent.parent == HERE.resolve()
+            and sha(args.inputs) == args.inputs_sha256, "own authenticated extraction manifest")
+    require(not args.binary.is_symlink() and args.binary.is_file()
+            and sha(args.binary) == args.binary_sha256, "root-reviewed fresh compiled binary")
+    cpus = sorted(os.sched_getaffinity(0))[:4]
+    require(len(cpus) == 4, "four available CPUs")
+    os.sched_setaffinity(0, cpus)
+    bindings = read_json(HERE / "bindings.json")
+    require(args.new_output.resolve().parent == HERE.resolve(), "fresh attempt within this ignored run")
+    args.new_output.mkdir(mode=0o700)
+    fd = os.open(args.binary.resolve(), os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
+    # Keep this exact executable FD through both execs. Check it before passing
+    # it across exec; no provider or unrelated process is touched.
+    try:
+        with os.fdopen(os.dup(fd), "rb") as opened:
+            import hashlib
+            require(hashlib.file_digest(opened, "sha256").hexdigest() == args.binary_sha256, "actual exec FD digest")
+        reports = {}
+        for model in ("0.8b", "2b"):
+            output = args.new_output / model
+            output.mkdir(mode=0o700)
+            before = source_files(bindings["models"][model])
+            row = execute(args.binary.resolve(), fd, args.inputs.resolve(), args.inputs_sha256,
+                          output.resolve(), model, cpus, freeze, before)
+            after = source_files(bindings["models"][model])
+            require(before == after, "all original source files unchanged after operators")
+            require(frozen() == freeze and sha(args.binary) == args.binary_sha256
+                    and sha(args.inputs) == args.inputs_sha256, "binary/source/input contracts unchanged")
+            write(output / "source-after.json", {"source_before": before, "source_after": after, "all_equal": True})
+            report = read_json(output / "report.json")
+            require(report["status"] == "PASS_PRIVATE_ATTENTION_FRAGMENT_ONLY"
+                    and report["inputs_sha256"] == args.inputs_sha256, "actual private numerical fragment passed")
+            reports[model] = {"report_sha256": sha(output / "report.json"), "invocation_sha256": sha(output / "invocation.json"),
+                              "source_after_sha256": sha(output / "source-after.json")}
+        write(args.new_output / "completed.json", {"status": "EXECUTED_REQUIRES_INDEPENDENT_BYTE_VALIDATION",
+            "models": reports, "freeze_manifest_sha256": freeze, "binary_sha256": args.binary_sha256,
+            "inputs": str(args.inputs.resolve()), "inputs_sha256": args.inputs_sha256,
+            "full_HF_or_StateRoot_or_GPU_or_performance_qualified": False})
+    finally:
+        os.close(fd)
+
+
+if __name__ == "__main__":
+    main()
diff --git a/.agents/runs/model-execution-original-attention-replay-20261020/test_contracts.py b/.agents/runs/model-execution-original-attention-replay-20261020/test_contracts.py
new file mode 100644
--- /dev/null
+++ b/.agents/runs/model-execution-original-attention-replay-20261020/test_contracts.py
@@ -0,0 +1,71 @@
+"""Prepared source-free rejection/transport fixtures; not trained expectations."""
+import json
+from pathlib import Path
+import struct
+import tempfile
+import unittest
+
+from contracts import archive, finite, payload, relative, source_2, source_8
+from validate_replay import cpus
+
+
+class Contracts(unittest.TestCase):
+    def test_transport_preserves_zero_sign_subnormals_and_finite_extremes(self):
+        words = (0, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f)
+        raw = struct.pack("<6H", *words)
+        finite(raw, 6)
+        with tempfile.TemporaryDirectory() as directory:
+            path = Path(directory) / "raw.safetensors"
+            header = {"empty": {"dtype": "I64", "shape": [0], "data_offsets": [0, 0]},
+                      "value": {"dtype": "BF16", "shape": [6], "data_offsets": [0, len(raw)]}}
+            encoded = json.dumps(header).encode()
+            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw)
+            actual = payload(path, archive(path)["value"], [6])
+            self.assertEqual(actual, raw)
+            self.assertEqual(struct.unpack("<6H", actual), words)
+
+    def test_nonfinite_and_partial_word_refusals(self):
+        for raw in (b"\0", struct.pack("<H", 0x7f80), struct.pack("<H", 0xff80), struct.pack("<H", 0x7fc1)):
+            with self.subTest(raw=raw), self.assertRaises(ValueError):
+                finite(raw, 1)
+
+    def test_archive_overlap_gap_truncation_and_duplicate_fields_refused(self):
+        with tempfile.TemporaryDirectory() as directory:
+            path = Path(directory) / "bad.safetensors"
+            headers = [
+                '{"a":{"dtype":"BF16","shape":[1],"data_offsets":[1,3]}}',
+                '{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}',
+                '{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}',
+                '{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}',
+            ]
+            for header in headers:
+                raw = header.encode()
+                path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0")
+                with self.subTest(header=header), self.assertRaises(ValueError):
+                    archive(path)
+
+    def test_wrong_shape_refuses_before_returning_bytes(self):
+        with self.assertRaises(ValueError):
+            payload(Path("never-opened"), {"dtype": "BF16", "shape": [2]}, [1])
+
+    def test_no_accepted_parent_no_source_claim(self):
+        for call in (lambda: source_8({"source_public_oracle_passed": False}, 0, {}),
+                     lambda: source_2({"status": "UNQUALIFIED"}, {}, 0, {})):
+            with self.assertRaises(ValueError):
+                call()
+
+    def test_exact_proc_affinity_is_not_just_a_cpu_count(self):
+        self.assertEqual(cpus("Cpus_allowed_list:\t0-1,4,7"), [0, 1, 4, 7])
+        self.assertNotEqual(cpus("Cpus_allowed_list:\t4-7"), [0, 1, 2, 3])
+        for invalid in ("wrong: 0-3", "Cpus_allowed_list: 3-0", "Cpus_allowed_list: 0-8192"):
+            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
+                cpus(invalid)
+
+    def test_external_path_escape_refused(self):
+        for invalid in ("", "/tmp/foreign", "../foreign", "a/../../foreign"):
+            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
+                relative(invalid)
+
+
+if __name__ == "__main__":
+    unittest.main()
diff --git a/.agents/runs/model-execution-original-attention-replay-20261020/validate_replay.py b/.agents/runs/model-execution-original-attention-replay-20261020/validate_replay.py
new file mode 100644
--- /dev/null
+++ b/.agents/runs/model-execution-original-attention-replay-20261020/validate_replay.py
@@ -0,0 +1,141 @@
+"""ROOT ONLY independent raw-byte comparisons; no numerical engine/imports."""
+import argparse
+import hashlib
+from pathlib import Path
+
+from contracts import (HERE, LABELS_8, PROFILE, SHAPES, STAGES, TOKENS, archive, finite,
+                       payload, pin, read_json, require, sha, source_2, source_8, write)
+from prepare_inputs import parents
+from run_replay import frozen
+
+
+def cpus(value):
+    fields = value.split(":", 1)
+    require(len(fields) == 2 and fields[0] == "Cpus_allowed_list", "actual proc affinity field")
+    result = []
+    for part in fields[1].strip().split(","):
+        ends = part.split("-")
+        require(1 <= len(ends) <= 2 and all(end.isdecimal() for end in ends), "actual CPU range syntax")
+        start, end = int(ends[0]), int(ends[-1])
+        require(0 <= start <= end < 4096, "bounded actual CPU range")
+        result.extend(range(start, end + 1))
+    return result
+
+
+def genuine_inputs(path):
+    binding = read_json(HERE / "bindings.json")
+    receipt8, paired, manifest = parents(binding)
+    reference = read_json(path)
+    require(reference["schema"] == "zerogravity.original-attention-fragment.raw-inputs.v1"
+            and reference["status"] == "AUTHENTIC_CPU_OPERATOR_INPUTS_ONLY_RUST_UNQUALIFIED"
+            and reference["profile"] == PROFILE and reference["layer"] == 3
+            and reference["teacher_tokens"] == list(TOKENS)
+            and reference["bindings_sha256"] == sha(HERE / "bindings.json")
+            and set(reference["models"]) == {"0.8b", "2b"}, "exact authentic extracted input contract")
+    root = path.parent
+    expected_files, total = {"inputs.json"}, 0
+    for model, item in binding["models"].items():
+        observed = reference["models"][model]
+        require(observed["source_repo"] == item["repo"] and observed["source_revision"] == item["revision"]
+                and observed["hidden"] == item["hidden"] and len(observed["cases"]) == 2,
+                "separate exact source identity/hidden width")
+        for step, frozen_case in enumerate(item["steps"]):
+            case = observed["cases"][step]
+            meta = read_json(pin(frozen_case["metadata"]["path"], frozen_case["metadata"]))
+            require(case["archive"] == frozen_case["archive"] and case["metadata"] == frozen_case["metadata"]
+                    and case["step"] == step and case["teacher_token"] == TOKENS[step]
+                    and case["profile"] == PROFILE and case["layer"] == 3
+                    and set(case["tensors"]) == {"hidden", *STAGES}, "exact full case/source binding")
+            source_8(receipt8, step, meta) if model == "0.8b" else source_2(paired, manifest, step, meta)
+            raw_archive = pin(frozen_case["archive"]["path"], frozen_case["archive"])
+            tensors = archive(raw_archive)
+            for stage in ("hidden",) + STAGES:
+                row = case["tensors"][stage]
+                shape = [1, 1, item["hidden"]] if stage == "hidden" else SHAPES[stage]
+                label = LABELS_8[stage] if model == "0.8b" else "layer-03." + stage
+                require(row["source_tensor_label"] == label and row["shape"] == shape and row["dtype"] == "BF16"
+                        and row["source_observer"] == meta["tensors"][label]
+                        and row["archive_offset"] == tensors[label]["offset"], "genuine source tensor and exact raw span")
+                genuine = payload(raw_archive, tensors[label], shape)
+                require(hashlib.sha256(genuine).hexdigest() == row["sha256"], "actual source payload digest")
+                name = f"{model}-step-{step}-{stage}.bf16"
+                require(row["file"] == name, "exclusive exact extraction filename")
+                extracted = pin(root / name, row).read_bytes()
+                require(extracted == genuine, "exported words equal authentic original archive, signed zeros included")
+                expected_files.add(name)
+                total += len(extracted)
+            pin(raw_archive, frozen_case["archive"])
+    require(total == reference["payload_bytes"] == 106496
+            and {p.name for p in root.iterdir()} == expected_files, "closed exact extract namespace and payload")
+    return reference
+
+
+def validate(output):
+    completed = read_json(output / "completed.json")
+    require(completed["status"] == "EXECUTED_REQUIRES_INDEPENDENT_BYTE_VALIDATION"
+            and completed["freeze_manifest_sha256"] == frozen(), "current root-built source closure")
+    path = Path(completed["inputs"])
+    require(path.resolve().parent.parent == HERE.resolve() and sha(path) == completed["inputs_sha256"],
+            "own immutable input extraction")
+    reference = genuine_inputs(path)
+    total = 0
+    for model, cases in reference["models"].items():
+        directory = output / model
+        pins = completed["models"][model]
+        for filename, key in (("report.json", "report_sha256"), ("invocation.json", "invocation_sha256"),
+                              ("source-after.json", "source_after_sha256")):
+            require(sha(directory / filename) == pins[key], "actual report/invocation/source result pin")
+        invoked, report = read_json(directory / "invocation.json"), read_json(directory / "report.json")
+        require(invoked["status"] == "CHILD_EXECUTED_REQUIRES_VALIDATION" and invoked["exit_code"] == 0
+                and invoked["reaped_exit_code"] == 0 and invoked["leader_reaped"] is True
+                and invoked["group_absent"] is True and invoked["cleanup_errors"] == []
+                and invoked["primary_error"] is None and invoked["pipe_eof"] is True
+                and invoked["pipe_closed"] is True and invoked["selector_closed"] is True
+                and invoked["log_closed"] is True and invoked["binary_sha256"] == completed["binary_sha256"]
+                and invoked["freeze_manifest_sha256"] == completed["freeze_manifest_sha256"], "complete actual owned lifecycle")
+        require(cpus(report["actual_affinity"]) == invoked["cpu_affinity"] and len(invoked["cpu_affinity"]) == 4,
+                "same exact four CPUs in executing native helper")
+        source = read_json(directory / "source-after.json")
+        require(source["all_equal"] is True and source["source_before"] == source["source_after"] == invoked["source_before"],
+                "all original source full-file digests and FD identities unchanged")
+        result = report["result"]
+        require(report["status"] == "PASS_PRIVATE_ATTENTION_FRAGMENT_ONLY"
+                and report["inputs_sha256"] == completed["inputs_sha256"] and result["model"] == model
+                and result["final_ledger"]["reserved"] == result["final_ledger"]["reservations"] == 0
+                and result["final_ledger"]["fenced"] is False and result["compared_output_words"] == 23552,
+                "pure numerical fragment and actual final reservations zero")
+        guards = result["guards"]
+        require(guards["pre_cancel"] is True and guards["wrong_layer"] is True
+                and guards["physical_pool_pressure"] is True and guards["failed_call_reservations_restored"] is True
+                and guards["mid_cancel_actual_projection_stage"]["per_class"]["TransferStaging"] > 0
+                and guards["mid_cancel_actual_projection_stage"]["per_class"]["Output"] >= 2 * 64 * 1024,
+                "real admission/cancel boundary and no failed private candidate leak")
+        for step, case in enumerate(cases["cases"]):
+            for dropped in (False, True):
+                comparison = read_json(directory / f"{model}-step-{step}-comparison-{str(dropped).lower()}.json")
+                require(comparison["different_words"] == 0 and comparison["after_callers_drop"] is dropped
+                        and comparison["teacher_token"] == TOKENS[step]
+                        and [v["stage"] for v in comparison["stages"]] == list(STAGES), "seven complete outputs before/after owner drop")
+            for stage in STAGES:
+                row = case["tensors"][stage]
+                actual = directory / f"{model}-step-{step}-{stage}.bf16"
+                # This compares actual Rust output to archive-authenticated raw
+                # BF16, independently of the Rust comparison's asserted result.
+                require(actual.stat().st_size == row["bytes"] and actual.read_bytes() == (path.parent / row["file"]).read_bytes(),
+                        "all actual raw BF16 output words exactly match original source")
+                finite(actual.read_bytes(), row["bytes"] // 2)
+                total += row["bytes"] // 2
+    require(total == 47104, "all four cases/seven outputs fully compared")
+    result = {"status": "PASS_ORIGINAL_CPU_ATTENTION_FRAGMENT_ONLY", "models": ["0.8b", "2b"],
+              "layer": 3, "profile": PROFILE, "teacher_tokens": list(TOKENS), "seven_output_words_exact": total,
+              "pressure_cancellation_and_retained_output_final_zero": True,
+              "full_HF_or_StateRoot_or_GPU_or_performance_qualified": False,
+              "completed_sha256": sha(output / "completed.json")}
+    write(output / "validation.json", result)
+    print(result)
+
+
+if __name__ == "__main__":
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument("--output", required=True, type=Path)
+    validate(parser.parse_args().output)

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/PRIOR_CHECKS_FOLLOWUP_V2.patch

Original UTF-8 bytes: 8189; SHA256 `1f5452e61339b0173f8ce7d7f2a7c475d98a6b2f146c0bc10238d086dd59fa79`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- README.md
+++ followup-v2/README.md
@@ -1,4 +1,4 @@
-# Root-only authentic CPU attention fragment replay
+# Root-only authentic CPU attention fragment replay — additive V2
 
 - Parent: existing model-execution plan. Source preparation only; root executes.
 - Current MAIN APIs only: sealed original `VerifiedTensorSource` →
@@ -9,6 +9,24 @@
   identity, BF16 matrices/learned QK gamma and epsilon1e-6 stay intact. All320
   text-role descriptors bind. Only `source-36-f32-restored` CPU AVX2 semantics;
   official all-BF16 loader and CUDA/vLLM mixed cast profile remain distinct.
+
+## V2 correction and preserved evidence
+
+- Original12 prepared files/frozen copies and root-inputs-1 failure stay unchanged.
+  Root found the original guard incorrectly compared a49-record metadata list
+  to scalar49. The parent report counts remain scalar98/49; per-step metadata
+  now requires the genuine full49 records, every bit_exact strictly true and
+  the complete original ordered fullV/cache label namespace exactly once.
+- No input/output/source/schema/math/shape/tolerance change. Rust/Cargo/bindings,
+  exporter,validator and owned-lifetime implementations are copied byte-exact.
+- Added metadata-only authentic49-record fixture and three regressions: genuine
+  list accepted; scalar/incomplete list refused; onebad/nonboolean/missing flag
+  or duplicate label refused. Root executes all10 fixtures; author AST only.
+- Reinspected remaining actual small .8B/2B metadata types: report scalarcounts,
+  replay_checks list, per-step tensor maps, public-call arguments,2B restored
+  source names36, profile/step/token fields and both observed lifecycle schemas.
+  No further metadata type mismatch found. No tensor payload read/hash or guard
+  execution by author. Fresh ROOT extract/replay directories are required.
 
 ## Independent reference and exact scope
 
@@ -65,7 +83,7 @@
 
 ```sh
 cd /home/pvl/spaces/abird/src/zerogravity
-RUN="$PWD/.agents/runs/model-execution-original-attention-replay-20261020"
+RUN="$PWD/.agents/runs/model-execution-original-attention-replay-20261020/followup-v2"
 # Source-free parser/transport fixtures; no trained qualification.
 PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$RUN" -p test_contracts.py -v
 # Genuine existing archive transport only; fresh directory directly under RUN.
--- TODO.md
+++ followup-v2/TODO.md
@@ -1,15 +1,13 @@
-# Working checklist
+# Additive V2 working checklist
 
 - Parent: existing model-execution plan; owner batch_design_review.
-- Owned scope: this new ignored run only. MAIN source/shared plans unchanged.
-- [x] Review narrow current MAIN attention compile/shape fixes; no blocker.
-- [x] Prepare same-engine authentic .8B/2B layer3 two-teacher seven-output harness.
-- [x] Keep genuine public source/schema/profile/archive and parent transparency binding.
-- [x] Prepare real same-source pool pressure,cancel,retained owner/final0 checks.
-- [x] Reuse exact accepted owned-native lifecycle functions; preserve original.
-- [ ] Independent source review,source-free fixtures,actual archive export.
-- [ ] ROOT build/current source+fresh binary binding and exact47,104-word replay.
-- [ ] ROOT strict/full phase checks and acceptance; no implementation promotion here.
+- Owned scope: fresh followup-v2 only; original12/frozen/failure preserved.
+- [x] Correct metadata prior_checks: actual49-record list, strict allbit_exact and exact full label cover; retain report scalar98/49.
+- [x] Add genuine metadata-only listfixture plus three regressions; original seven fixtures unchanged.
+- [x] Inspect remaining genuine small metadata schema/types without tensor payload/math/runtime.
+- [x] Copy all other Rust/Cargo/bindings/export/validator/owned cleanup bytes unchanged.
+- [ ] Independent ROOT/peer source review and10 sourcefree fixtures.
+- [ ] ROOT fresh authenticated export, build/binary/source binding,47,104-word replay and independent byte validation.
 
-Author: AST/source inspection only. No tests,compilation,model/operator/GPU or
-container execution; no large raw/model/DSO read/hash. No physical result claimed.
+Original actual root-inputs-1 export failure remains unqualified/preserved. No
+author build/test/operator/model/GPU/container run or tensor/provider hash.
--- contracts.py
+++ followup-v2/contracts.py
@@ -123,6 +123,18 @@
     return raw
 
 
+def prior_checks_8(records):
+    require(type(records) is list and len(records) == 49,
+            "actual .8B prior checks are 49 individual records, never a scalar count")
+    labels = ["logits.full248320"]
+    for layer in range(24):
+        labels.extend(f"layer-{layer:02d}.{domain}" for domain in
+                      (("keys", "values") if layer % 4 == 3 else ("conv", "recurrent")))
+    require(all(type(row) is dict and row.get("bit_exact") is True for row in records)
+            and [row.get("label") for row in records] == labels,
+            "all49 original fullV/cache records present once in source order and bit-exact")
+
+
 def source_8(report, step, metadata):
     require(report["source_public_oracle_passed"] is True and report["rust_qualified"] is False
             and report["actual_attention_calls"] == 12 and report["actual_rotary_calls"] == 2
@@ -133,7 +145,9 @@
             and report["source_receipt"]["repo"] == "Qwen/Qwen3.5-0.8B"
             and report["source_receipt"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
             and report["steps"][step]["token"] == TOKENS[step]
-            and metadata["prior_checks"] == 49, "actual accepted .8B source/runtime/complete transparency")
+            and report["steps"][step]["prior_checks"] == 49,
+            "actual accepted .8B source/runtime/complete transparency")
+    prior_checks_8(metadata["prior_checks"])
     call = metadata["calls"][1]
     require(call["kind"] == "attention" and call["layer"] == 3 and call["prefix"] == "call001"
             and call["hidden"] == LABELS_8["hidden"] and call["inference_mode"] is True,
--- test_contracts.py
+++ followup-v2/test_contracts.py
@@ -1,15 +1,43 @@
 """Prepared source-free rejection/transport fixtures; not trained expectations."""
 import json
+import copy
 from pathlib import Path
 import struct
 import tempfile
 import unittest
 
-from contracts import archive, finite, payload, relative, source_2, source_8
+from contracts import archive, finite, payload, prior_checks_8, relative, source_2, source_8
 from validate_replay import cpus
 
 
 class Contracts(unittest.TestCase):
+    def test_actual_observed_prior_check_list_contract(self):
+        fixture = json.loads((Path(__file__).parent / "prior-checks-observed.json").read_text())
+        prior_checks_8(fixture["prior_checks"])
+        self.assertEqual(len(fixture["prior_checks"]), 49)
+        self.assertEqual(fixture["prior_checks"][0]["label"], "logits.full248320")
+
+    def test_prior_scalar_count_and_missing_records_refused(self):
+        fixture = json.loads((Path(__file__).parent / "prior-checks-observed.json").read_text())
+        for invalid in (49, True, {"count": 49}, fixture["prior_checks"][:-1]):
+            with self.subTest(invalid=type(invalid).__name__), self.assertRaises(ValueError):
+                prior_checks_8(invalid)
+
+    def test_every_prior_check_must_be_exact_and_unique(self):
+        fixture = json.loads((Path(__file__).parent / "prior-checks-observed.json").read_text())
+        for mutation in ("false", "integer", "missing", "duplicate"):
+            rows = copy.deepcopy(fixture["prior_checks"])
+            if mutation == "false":
+                rows[24]["bit_exact"] = False
+            elif mutation == "integer":
+                rows[24]["bit_exact"] = 1
+            elif mutation == "missing":
+                del rows[24]["bit_exact"]
+            else:
+                rows[24] = copy.deepcopy(rows[23])
+            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
+                prior_checks_8(rows)
+
     def test_transport_preserves_zero_sign_subnormals_and_finite_extremes(self):
         words = (0, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f)
         raw = struct.pack("<6H", *words)

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/STEADY_LEDGER_FOLLOWUP_V3.patch

Original UTF-8 bytes: 3088; SHA256 `50268607a674f918e69a02c35b1fe1a7d6d4fe2e41a8dd0df421c08d2ea45689`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- followup-v2/main.rs
+++ followup-v3/main.rs
@@ -470,8 +470,12 @@
     )?;
     let before = snapshot(&host)?;
     ensure(
-        before.reserved == SOURCE_META + 4 * MIB + CONTROL
-            && class(&before, AllocationClass::Output) == 0,
+        before.reserved == SOURCE_META + 8 * MIB + CONTROL
+            && before.reservations == 4
+            && class(&before, AllocationClass::Scratch) == SOURCE_META + 8 * MIB
+            && class(&before, AllocationClass::Other) == CONTROL
+            && class(&before, AllocationClass::Output) == 0
+            && class(&before, AllocationClass::TransferStaging) == 0,
         "source/binding/control steady exact reservation",
     )?;
     let cases = references["cases"]
@@ -558,7 +562,8 @@
     )?;
     Ok(
         json!({"model":model,"source_identity":format!("{identity:?}"),"guards":guard_report,
-        "cases":results,"retained_after_callers_drop":ledger(&retained),"final_ledger":ledger(&final_row),
+        "cases":results,"steady_ledger":ledger(&before),
+        "retained_after_callers_drop":ledger(&retained),"final_ledger":ledger(&final_row),
         "compared_output_words":compared_words,"source_window_payloads_immutable":true}),
     )
 }
--- followup-v2/README.md
+++ followup-v3/README.md
@@ -83,7 +83,7 @@
 
 ```sh
 cd /home/pvl/spaces/abird/src/zerogravity
-RUN="$PWD/.agents/runs/model-execution-original-attention-replay-20261020/followup-v2"
+RUN="$PWD/.agents/runs/model-execution-original-attention-replay-20261020/followup-v3"
 # Source-free parser/transport fixtures; no trained qualification.
 PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$RUN" -p test_contracts.py -v
 # Genuine existing archive transport only; fresh directory directly under RUN.
@@ -110,3 +110,15 @@
 RoPE,KV/cache append,BMM/softmax/gating integration,initial/cached GDN coefficients
 and MKL scheduling,full-HF session/checkpoint/logits,27B,GPU/native cuBLAS and
 matched serving performance remain unqualified.
+
+## ROOT additive V3 accounting correction
+
+- V2 root-replay-2 refused before numerical operators: steady charge omitted
+  QwenOriginalWindowText's4MiB metadata. OriginalWindowOperators retains a
+  separate4MiB metadata charge. Exact steady total is64+8+2MiB/4reservations.
+- Assert exact Scratch72MiB,Other2MiB,Output/Staging0; retain full final0 and
+  every pressure/cancellation/lifetime/numerical guard. Add actual steady ledger
+  to report. Both4MiB binding charges drop with callers; retained output/source
+  charge remains64+2+16MiB. No MAIN/math/input/reference changes.
+- OriginalV2 sources/frozen binary/failed invocation/report/log remain intact.
+- Runtime status: ROOT must rebuild and execute; correction is not qualification.
--- followup-v2/TODO.md
+++ followup-v3/TODO.md
@@ -11,3 +11,5 @@
 
 Original actual root-inputs-1 export failure remains unqualified/preserved. No
 author build/test/operator/model/GPU/container run or tensor/provider hash.
+
+- [ ] ROOT V3: rebuild exact accounting correction, rerun both trained fragments and independent validation.

````

