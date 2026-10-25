"""Streaming expected-only raw transport; no source/framework imports at import."""
import sys
from contract import require, digest, extent, finite, state_specs


class Archive:
    def __init__(self, path, plan, torch):
        self.stream = path.open('xb')
        self.plan, self.torch = plan, torch
        self.records = []
        self.total = 0
        self.tag = None

    def add(self, label, layer, value):
        t = self.torch
        require(self.tag is not None and isinstance(value, t.Tensor) and value.device.type == 'cpu'
                and value.layout == t.strided and sys.byteorder == 'little', 'actual CPU strided observation')
        dtype = {t.bfloat16: 'BF16', t.float32: 'F32', t.int64: 'I64'}.get(value.dtype)
        require(dtype is not None, 'actual original observation codec')
        shape = list(value.shape)
        size = extent(shape, dtype)
        require(0 < size <= self.plan['record_cap'] and self.total+size <= self.plan['archive_cap'],
                'record/archive admission BEFORE transport')
        integer = t.int16 if dtype == 'BF16' else t.int32 if dtype == 'F32' else t.int64
        raw = value.detach().view(integer).numpy().tobytes(order='C')
        require(len(raw) == size and finite(raw, dtype), 'complete finite actual observation')
        self.stream.write(raw)
        self.records.append({**self.tag, 'layer': layer, 'label': label, 'shape': shape,
            'original_stride': list(value.stride()), 'dtype': dtype, 'device': 'cpu',
            'offset': self.total, 'bytes': size, 'sha256': digest(raw)})
        self.total += size

    def states(self, cache, moment):
        specs = state_specs(self.tag['call'], moment)
        for layer, label, shape, dtype in specs:
            state = cache.layers[layer]
            value = (state.conv_states[0] if label == 'conv4' else state.recurrent_states[0]
                     if label == 'recurrent' else state.keys if label == 'key' else state.values)
            require(list(value.shape) == shape and str(value.dtype) == {'BF16': 'torch.bfloat16', 'F32': 'torch.float32'}[dtype],
                    'full original Conv4/key-majorFP32/head-majorKV cache')
            self.add(moment+'.'+label, layer, value)

    def close(self):
        self.stream.flush()
        self.stream.close()
