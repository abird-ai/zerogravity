"""Original public HF RMS and Torch embedding/linear, no model evaluator."""
import os
import struct
from pathlib import Path
from contract import MODEL_SHA, STAGES, TORCH_GIT, digest, fixture, metadata, parse, read, require, write


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    require(resource.getrlimit(resource.RLIMIT_NOFILE)==(1024,1024), 'bounded original FDs')
    import torch
    import transformers
    from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
    from transformers.models.qwen3_5 import modeling_qwen3_5 as original
    require(torch.__version__ == '2.13.0+cu129' and torch.version.git_version == TORCH_GIT
            and transformers.__version__ == '5.17.0', 'pinned approved original runtime')
    require(digest(read(Path(original.__file__))) == MODEL_SHA, 'actual original HF source')
    cpus = [int(v) for v in os.environ['ZG_ALLOWED_CPUS'].split(',')]
    require(len(cpus) == len(set(cpus)) == 4, 'four explicit original CPUs')
    os.sched_setaffinity(0, cpus)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    require(torch.backends.cpu.get_cpu_capability() == 'AVX2' and not torch.cuda.is_initialized(), 'original CPU AVX2/no CUDA context')
    require(torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1, 'bounded CPU threads')
    plan = metadata('/output/input.json')
    width, rows, token = plan['width'], plan['rows'], plan['token']
    source = read('/source/source.safetensors')
    require(source == fixture(width, rows) and digest(source) == plan['source_sha256'] and token == 16, 'authentic synthetic source')
    header_bytes, = struct.unpack_from('<Q', source)
    from contract import unique
    import json
    header = json.loads(source[8:8 + header_bytes], object_pairs_hook=unique)
    tensors = {}
    for name in ('embedding', 'gamma', 'projection'):
        spec = header[name]
        begin, end = spec['data_offsets']
        payload = bytearray(source[8 + header_bytes + begin:8 + header_bytes + end])
        tensor = torch.frombuffer(payload, dtype=torch.bfloat16).clone().reshape(spec['shape'])
        require(torch.isfinite(tensor).all().item(), 'finite raw source tensor')
        tensors[name] = tensor
    observed = {}

    def remember(label, tensor):
        require(label not in observed, 'exact one original operation per observed boundary')
        with _disable_current_modes():
            observed[label] = tensor.detach().clone()

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, op, types, args=(), kwargs=None):
            result = op(*args, **(kwargs or {}))
            name = str(op)
            cast = name in ('aten._to_copy.default', 'aten.to.dtype', 'aten.type_as.default')
            if cast and result.dtype == torch.float32:
                remember('embedding' if result.shape == (1, width) else 'raw_gamma', result)
            elif name == 'aten.pow.Tensor_Scalar':
                require(args[1] == 2, 'original square exponent')
                remember('squares', result)
            elif name == 'aten.mean.dim':
                remember('mean', result)
            elif name == 'aten.add.Tensor':
                remember('denominator' if result.shape == (1, 1) else 'runtime_gamma', result)
            elif name == 'aten.rsqrt.default':
                remember('inverse', result)
            elif name == 'aten.mul.Tensor':
                remember('normalized' if 'normalized' not in observed else 'weighted', result)
            elif cast and result.dtype == torch.bfloat16:
                remember('rms_bf16', result)
            else:
                raise ValueError('unrecognized original RMS dispatch: ' + name)
            return result

    with torch.inference_mode():
        layer = original.Qwen3_5RMSNorm(width, eps=1e-6).eval()
        layer.weight.copy_(tensors['gamma'])
        # Keep learned source dtype; module defaults must not widen raw gamma.
        layer.to(dtype=torch.bfloat16)
        actual_input = torch.nn.functional.embedding(torch.tensor([token], dtype=torch.int64), tensors['embedding'])
        with Trace():
            normalized = layer(actual_input)
        observed['projection_bf16'] = torch.nn.functional.linear(normalized, tensors['projection'])
    require(set(observed) == set(STAGES), 'all original source/materialized boundaries observed')
    raw = bytearray(b'ZGBF16V1' + bytes.fromhex(plan['source_sha256']) + struct.pack('<4I', width, rows, token, len(STAGES)))
    for label in STAGES:
        tensor = observed[label].float().contiguous()
        payload = tensor.view(torch.uint8).reshape(-1).numpy().tobytes()
        raw.extend(struct.pack('<I', len(payload)))
        raw.extend(payload)
    records = parse(raw, plan['source_sha256'], width, rows, token)
    require(read('/source/source.safetensors') == source and not torch.cuda.is_initialized(), 'same original source/no CUDA after math')
    Path('/output/reference.bin').write_bytes(raw)
    write('/output/capture.json', {'schema': 'zg.portable-bf16-original.v1', 'status': 'ORIGINAL_REFERENCE_REQUIRES_VALIDATION',
        'torch': torch.__version__, 'torch_git': torch.version.git_version, 'transformers': transformers.__version__,
        'modeling_sha256': MODEL_SHA, 'cpu_capability': 'AVX2', 'address_space_limit': list(resource.getrlimit(resource.RLIMIT_AS)), 'fd_limit': list(resource.getrlimit(resource.RLIMIT_NOFILE)), 'affinity': sorted(os.sched_getaffinity(0)), 'threads': 4, 'interop_threads': 1, 'source_sha256': plan['source_sha256'],
        'width': width, 'rows': rows, 'token': token, 'records': records, 'bytes': len(raw), 'sha256': digest(raw),
        'scope': {'synthetic_source': True, 'trained_model': False, 'portable_arithmetic_qualified': False,
                  'runtime_rust_source_in_original': False, 'state_root': False, 'serving': False, 'performance': False}})
    print('ORIGINAL_PORTABLE_BF16_SLICE_COMPLETE', width, rows, len(raw), flush=True)


if __name__ == '__main__':
    main()
