"""ROOT-only original full text forward, with bounded direct source bindings."""
import importlib.metadata
import json
import os
from pathlib import Path
import resource
from contract import (HERE, MODEL_SHA, TORCH_GIT, KINDS, require, metadata, package,
                      specs, digest, sha, write, record_specs, scope)
from source_owner import SourceOwner
from archive import Archive

OWNER = None


def cache_flags(cache, empty):
    require(len(cache.layers) == 64, 'original64 cache layer owners')
    rows = []
    for i, kind in enumerate(KINDS):
        s = cache.layers[i]
        if kind == 'linear_attention':
            conv = bool(s.is_conv_states_initialized[0])
            recurrent = bool(s.is_recurrent_states_initialized[0])
            previous = bool(s.has_previous_state[0])
            require(conv is recurrent is previous is (not empty) and s.record_past is False,
                    'actual original Conv4/recurrent initialization flags')
            rows.append({'layer': i, 'kind': kind, 'conv': conv, 'recurrent': recurrent, 'previous': previous})
        else:
            key = getattr(s, 'keys', None) is not None
            value = getattr(s, 'values', None) is not None
            require(key is value is (not empty), 'actual original independent KV initialization')
            rows.append({'layer': i, 'kind': kind, 'key': key, 'value': value})
    return rows


def main():
    global OWNER
    out = Path('/output/captured')
    require(not out.exists(), 'fresh original full reference')
    out.mkdir(mode=0o700)
    report = {'schema': 'zg.original27b-full-window-capture.v1', 'status': 'UNQUALIFIED', 'scope': scope()}
    archive = None
    try:
        cpus = [int(x) for x in os.environ['ZG_ALLOWED_CPUS'].split(',')]
        require(len(cpus) == len(set(cpus)) == 4, 'four exact CPUs')
        os.sched_setaffinity(0, cpus)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' and os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void'
                and os.environ.get('ATEN_CPU_CAPABILITY') == 'avx2' and not os.environ.get('LD_PRELOAD')
                and not os.environ.get('LD_AUDIT') and os.environ.get('USE_HUB_KERNELS') == '0', 'isolated public CPU reference')
        initial = package()
        require(digest(json.dumps(initial, sort_keys=True).encode()) == os.environ['ZG_PACKAGE_SHA'], 'worker package')
        plan = metadata(HERE/'inputs.json')
        pins = metadata(HERE/'source-pin.json')
        specs(plan, pins)
        resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
        OWNER = SourceOwner(plan, pins)
        import torch
        from transformers.models.qwen3_5 import modeling_qwen3_5 as original
        from transformers import cache_utils
        require(torch.__version__ == '2.13.0+cu129' and torch.version.git_version == TORCH_GIT
                and importlib.metadata.version('transformers') == '5.17.0'
                and sha(original.__file__) == MODEL_SHA and sha(cache_utils.__file__) == plan['cache_utils_sha256']
                and not torch.cuda.is_available(), 'pinned original code and CPU-only library')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        require(torch.backends.cpu.get_cpu_capability() == 'AVX2', 'actual ATen profile; MKL remains original')
        config = original.Qwen3_5TextConfig(**OWNER.config)
        config._attn_implementation = 'eager'
        with torch.device('meta'):
            model = original.Qwen3_5ForCausalLM(config)
        # Pure original rotary construction needs actual CPU frequency buffers;
        # learned851 parameters remain meta until their own public module runs.
        model.model.rotary_emb = original.Qwen3_5TextRotaryEmbedding(config, device='cpu')
        model.eval()
        archive = Archive(out/'raw.bin', plan, torch)
        tag, order, hooks = {}, [], []
        groups = [('embedding', model.model.embed_tokens, 'embedding', None)]
        groups += [('layer.'+str(i), module, 'layer_output', i) for i, module in enumerate(model.model.layers)]
        groups += [('final', model.model.norm, 'normalized', None), ('head', model.lm_head, 'logits', None)]
        expected_groups = [g[0] for g in groups]
        require(len(expected_groups) == 67, 'original full decoder/normalizer/head module cohort')
        def before(group):
            def load(module, _args):
                require(group == expected_groups[len(order)], 'same original full forward traversal order')
                OWNER.bind(group, module, torch, tag)
                order.append(group)
            return load
        def after(group, label, layer):
            def save(module, _args, value):
                archive.add(label, layer, value)
                OWNER.release(group, module)
            return save
        for group, module, label, layer in groups:
            hooks.append(module.register_forward_pre_hook(before(group)))
            hooks.append(module.register_forward_hook(after(group, label, layer)))
        def positions(_module, args):
            require(len(args) == 2 and list(args[1].shape) == [3, 1, 1], 'original four-axis split to three-axis rotary')
            archive.add('positions', None, args[1])
        hooks.append(model.model.rotary_emb.register_forward_pre_hook(positions))
        calls = []
        with torch.inference_mode():
            for repeat in range(2):
                cache = cache_utils.DynamicCache(config=config)
                for call, teacher in enumerate(plan['teachers']):
                    tag.clear()
                    tag.update(repeat=repeat, call=call, teacher=teacher, position=call)
                    archive.tag = tag.copy()
                    order.clear()
                    before_flags = cache_flags(cache, call == 0)
                    require(cache.get_seq_length() == call, 'actual cache frontier before original model call')
                    archive.states(cache, 'before')
                    result = model(input_ids=torch.tensor([[teacher]], dtype=torch.long),
                        past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=0)
                    require(order == expected_groups and OWNER.active is None
                            and result.past_key_values is cache and cache.get_seq_length() == call+1,
                            'one complete original64 forward with same cache and released source group')
                    after_flags = cache_flags(cache, False)
                    archive.states(cache, 'after')
                    calls.append({**tag, 'before': before_flags, 'after': after_flags,
                        'original_module_order': order.copy(), 'cache_frontier': cache.get_seq_length(),
                        'greedy_id_diagnostic_only': int(result.logits[0, 0].argmax().item())})
                    del result
                del cache
        for hook in hooks:
            hook.remove()
        expected = list(record_specs())
        require(len(archive.records) == len(expected) == plan['expected_records']
                and archive.total == plan['expected_raw_bytes'], 'complete declared full state/producer archive')
        archive.close()
        OWNER.finish()
        require(package() == initial, 'same worker package after full original forward')
        report.update(status='CAPTURED_ORIGINAL27B_FULL_WINDOW_REQUIRES_VALIDATION', package=initial,
            snapshot=plan['snapshot'], repository=plan['repository'], revision=plan['revision'],
            parameter_mode=plan['parameter_mode'], records=archive.records, calls=calls,
            source_readsets=OWNER.readsets, source_counts=OWNER.counts, source_views=OWNER.views,
            source_FDs=OWNER.before, source_events=OWNER.events, maximum_mapped_bytes=OWNER.max_mapping_bytes,
            all_original_source_and_mapped_bytes_unchanged=True, source_owner_retained_until_process_exit=True,
            raw={'file': 'raw.bin', 'bytes': archive.total, 'sha256': sha(out/'raw.bin')},
            torch=torch.__version__, torch_git=torch.version.git_version, modeling_sha256=MODEL_SHA,
            cache_utils_sha256=plan['cache_utils_sha256'], affinity=sorted(os.sched_getaffinity(0)),
            threads=4, interop_threads=1, original_full_forward=True, cache_domains=112,
            full_hf_conv_width=4, all_851_source_parameters_raw=True)
    except BaseException as exc:
        report['failure'] = repr(exc)
        raise
    finally:
        if archive is not None and not archive.stream.closed:
            archive.close()
        write(out/'manifest.json', report)


if __name__ == '__main__':
    main()
