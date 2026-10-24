"""Finite full64 original reference protocol. Import never loads a framework."""
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

HERE = Path(__file__).resolve().parent
IMAGE = 'sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe'
IMAGE_DIGEST = 'sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634'
MODEL_SHA = '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b'
CACHE_SHA = '702144bb44553f6339ea1bf23c8205a708bb5f8c7c09cb3a2db484182646743c'
TORCH_GIT = 'cf30153c4c131c8164ee7798e5022d810682e2cb'
REGISTRY_SHA = '2bb924d3c486e14b66cc43f9314000970b627f37a53dc8fc08e8f08079e68a2f'
KINDS = tuple('full_attention' if i % 4 == 3 else 'linear_attention' for i in range(64))
FILES = ('contract.py', 'capture.py', 'source_owner.py', 'archive.py', 'run_reference.py',
         'validate_reference.py', 'owned_container_v4.py', 'inputs.json', 'source-pin.json', 'full-reference-pin.json')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(m):
    return [m.st_dev, m.st_ino, m.st_size, m.st_mtime_ns, m.st_ctime_ns]


def unique(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'duplicate JSON key')
        out[key] = value
    return out


def fd_sha(fd, offset, count):
    h = hashlib.sha256()
    for i in range(0, count, 1024**2):
        n = min(1024**2, count-i)
        raw = os.pread(fd, n, offset+i)
        require(len(raw) == n, 'complete positioned source/reference read')
        h.update(raw)
    return h.hexdigest()


def read(path, cap=16*1024**2, pin=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular file')
        with os.fdopen(os.dup(fd), 'rb') as f:
            raw = f.read(cap+1)
        require(len(raw) == before.st_size and fingerprint(before) == fingerprint(os.fstat(fd)), 'held bytes/EOF')
        require(fingerprint(os.stat(path, follow_symlinks=False)) == fingerprint(before), 'same current path')
    finally:
        os.close(fd)
    if pin is not None:
        require(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'whole content pin')
    return raw


def metadata(path, cap=16*1024**2, pin=None):
    return json.loads(read(path, cap, pin), object_pairs_hook=unique)


def sha(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), 'regular package/source identity')
        result = fd_sha(fd, 0, before.st_size)
        require(os.pread(fd, 1, before.st_size) == b''
                and fingerprint(os.fstat(fd)) == fingerprint(before)
                and fingerprint(os.stat(path, follow_symlinks=False)) == fingerprint(before),
                'same complete package/source FD/path after scan')
        return result
    finally:
        os.close(fd)


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    require(len(raw) <= 16*1024**2, 'metadata output bound')
    with Path(path).open('xb') as f:
        os.fchmod(f.fileno(), 0o600)
        f.write(raw)


def package():
    return {name: {'bytes': (HERE/name).stat().st_size, 'sha256': sha(HERE/name)} for name in FILES}


def full_scope():
    return {'original_full_text_forward': True, 'all_851_text_source_roles': True,
            'full_hf_conv4': True, 'mixed_source_parameters_preserved': True,
            'teacher_forcing_only': True, 'natural_feedback_qualified': False,
            'rust_numerical_parity_qualified': False, 'state_root_native_serving_performance_qualified': False,
            'all_bf16_loader_equivalence_qualified': False, 'mkl_exp_producer_qualified': False}


def specs(plan, pins):
    require(digest(json.dumps(pins, sort_keys=True).encode()) == REGISTRY_SHA, 'approved complete source registry')
    require(plan['schema'] == 'zg.original27b-full-window-inputs.v1'
            and plan['snapshot'] == pins['snapshot'] == '98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e'
            and plan['repository'] == pins['repository'] == 'Qwen/Qwen3.8-27B'
            and plan['revision'] == pins['revision'] == '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0', 'exact source namespace')
    require(plan['config_sha256'] == '191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab'
            and plan['index_sha256'] == '77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df', 'original config/index')
    require(plan['source_directory'] == pins['reference_directory'] and plan['cache_utils_sha256'] == CACHE_SHA, 'fixed approved source and original cache code')
    require(plan['teachers'] == [11, 353] and plan['repetitions'] == 2
            and plan['parameter_mode'] == 'raw-source-assign-no-cast' and plan['scope'] == full_scope(), 'finite producer scope')
    require(plan['source_scan_bytes'] == 1024**2 and plan['mapping_cap'] == 3*1024**3
            and plan['record_cap'] == 4*1024**2 and plan['archive_cap'] == 1024**3
            and plan['expected_records'] == 1040 and plan['expected_raw_bytes'] == 934776928, 'closed allocation/transport limits')
    files = {r['name']: r for r in pins['files']}
    require(len(files) == 31 and sum(r['bytes'] for n, r in files.items() if n.endswith('.safetensors')) == 55563006776, 'whole approved source cohort')
    return files


def roles():
    rows = []
    def add(group, local, shape, matrix=False, layer=None):
        name = ('lm_head.weight' if group == 'head' else 'model.language_model.' +
                (('layers.'+str(layer)+'.') if layer is not None else '') + local)
        rows.append({'group': group, 'layer': layer, 'local': local, 'name': name,
                     'shape': shape, 'allowed_dtypes': ['BF16'] if matrix else ['BF16', 'F32']})
    add('embedding', 'embed_tokens.weight', [248320, 5120], True)
    for i, kind in enumerate(KINDS):
        g = 'layer.'+str(i)
        for n in ('input_layernorm.weight', 'post_attention_layernorm.weight'):
            add(g, n, [5120], layer=i)
        for n, shape in [('gate', [17408, 5120]), ('up', [17408, 5120]), ('down', [5120, 17408])]:
            add(g, 'mlp.'+n+'_proj.weight', shape, True, i)
        if kind == 'linear_attention':
            for n, shape, matrix in [('in_proj_qkv.weight', [10240, 5120], True),
                    ('in_proj_z.weight', [6144, 5120], True), ('in_proj_a.weight', [48, 5120], True),
                    ('in_proj_b.weight', [48, 5120], True), ('out_proj.weight', [5120, 6144], True),
                    ('conv1d.weight', [10240, 1, 4], True), ('A_log', [48], False),
                    ('dt_bias', [48], False), ('norm.weight', [128], False)]:
                add(g, 'linear_attn.'+n, shape, matrix, i)
        else:
            for n, shape in [('q', [12288, 5120]), ('k', [1024, 5120]), ('v', [1024, 5120]), ('o', [5120, 6144])]:
                add(g, 'self_attn.'+n+'_proj.weight', shape, True, i)
            for n in ('q_norm', 'k_norm'):
                add(g, 'self_attn.'+n+'.weight', [256], layer=i)
    add('final', 'norm.weight', [5120])
    add('head', 'weight', [248320, 5120], True)
    require(len(rows) == len({r['name'] for r in rows}) == 851, 'entire text role namespace')
    return rows


def require_config(c):
    require(c['hidden_size'] == 5120 and c['intermediate_size'] == 17408 and c['vocab_size'] == 248320
            and c['num_hidden_layers'] == 64 and tuple(c['layer_types']) == KINDS
            and c['num_attention_heads'] == 24 and c['num_key_value_heads'] == 4 and c['head_dim'] == 256
            and c['linear_num_key_heads'] == 16 and c['linear_num_value_heads'] == 48
            and c['linear_key_head_dim'] == c['linear_value_head_dim'] == 128 and c['linear_conv_kernel_dim'] == 4
            and c['rms_norm_eps'] == 1e-6 and c['hidden_act'] == 'silu' and c['tie_word_embeddings'] is False,
            'genuine64/groupedH48/text geometry')


def source_header(fd, pin):
    before = os.fstat(fd)
    require(stat.S_ISREG(before.st_mode) and before.st_size == pin['bytes'], 'held source extent')
    raw = os.pread(fd, 8, 0)
    require(len(raw) == 8, 'source header prefix')
    n, = struct.unpack('<Q', raw)
    require(1 < n <= 8*1024**2 and n+8 <= before.st_size, 'bounded source header')
    raw = os.pread(fd, n, 8)
    require(len(raw) == n and fingerprint(os.fstat(fd)) == fingerprint(before), 'held complete header')
    return n+8, json.loads(raw, object_pairs_hook=unique)


def role_view(start, header, role, file_bytes):
    row = header[role['name']]
    n = 1
    for dim in role['shape']:
        require(type(dim) is int and dim > 0, 'positive exact role shape')
        n *= dim
    require(row['dtype'] in role['allowed_dtypes'] and row['shape'] == role['shape'], 'raw original dtype/shape')
    size = n * (2 if row['dtype'] == 'BF16' else 4)
    a, b = row['data_offsets']
    require(type(a) is int and type(b) is int and 0 <= a < b and b-a == size and start+b <= file_bytes, 'full source interval')
    absolute = start+a
    require(os.sysconf('SC_PAGE_SIZE') == 4096, 'reviewed mapping page')
    aligned = absolute-absolute % 4096
    return {'absolute': absolute, 'bytes': size, 'aligned': aligned, 'view_offset': absolute-aligned,
            'mapping_bytes': size+absolute-aligned, 'dtype': row['dtype'], 'shape': row['shape']}


def full_state_specs(position, moment):
    require(type(position) is int and position in (0, 1) and moment in ('before', 'after'), 'finite actual call/state moment')
    if moment == 'before' and position == 0:
        return []
    count = position if moment == 'before' else position+1
    out = []
    for i, kind in enumerate(KINDS):
        if kind == 'linear_attention':
            out += [(i, 'conv4', [1, 10240, 4], 'BF16'), (i, 'recurrent', [1, 48, 128, 128], 'F32')]
        else:
            out += [(i, 'key', [1, 4, count, 256], 'BF16'), (i, 'value', [1, 4, count, 256], 'BF16')]
    return out


def full_record_specs():
    for repeat in range(2):
        for call, teacher in enumerate((11, 353)):
            for layer, label, shape, dtype in state_specs(call, 'before'):
                yield repeat, call, teacher, layer, 'before.'+label, shape, dtype
            yield repeat, call, teacher, None, 'embedding', [1, 1, 5120], 'BF16'
            yield repeat, call, teacher, None, 'positions', [3, 1, 1], 'I64'
            for layer in range(64):
                yield repeat, call, teacher, layer, 'layer_output', [1, 1, 5120], 'BF16'
            yield repeat, call, teacher, None, 'normalized', [1, 1, 5120], 'BF16'
            yield repeat, call, teacher, None, 'logits', [1, 1, 248320], 'BF16'
            for layer, label, shape, dtype in state_specs(call, 'after'):
                yield repeat, call, teacher, layer, 'after.'+label, shape, dtype


def extent(shape, dtype):
    n = 1
    for d in shape:
        require(type(d) is int and d > 0, 'positive record dimensions')
        n *= d
    return n*{'BF16': 2, 'F32': 4, 'I64': 8}[dtype]


def record_metadata(row, expected, offset):
    repeat, call, teacher, layer, label, shape, dtype = expected
    n = extent(shape, dtype)
    require(all(type(row[k]) is int for k in ('repeat', 'call', 'teacher', 'position', 'offset', 'bytes'))
            and (row['layer'] is None if layer is None else type(row['layer']) is int)
            and isinstance(row['shape'], list) and all(type(d) is int for d in row['shape'])
            and row['repeat'] == repeat and row['call'] == call and row['teacher'] == teacher
            and row['position'] == call and row['layer'] == layer and row['label'] == label
            and row['shape'] == shape and row['dtype'] == dtype and row['device'] == 'cpu'
            and row['offset'] == offset and row['bytes'] == n,
            'exact original producer/state order/shape/codec/coverage')
    stride = row['original_stride']
    elements = n // {'BF16': 2, 'F32': 4, 'I64': 8}[dtype]
    require(isinstance(stride, list) and len(stride) == len(shape)
            and all(type(s) is int and s >= 0 for s in stride)
            and sum((d-1)*s for d, s in zip(shape, stride)) < elements,
            'bounded original strides in elements, including position expansion')
    return n


def finite(raw, dtype):
    if dtype == 'I64':
        return len(raw) % 8 == 0
    width, mask, fmt = (2, 0x7f80, '<H') if dtype == 'BF16' else (4, 0x7f800000, '<I')
    return len(raw) % width == 0 and all(w & mask != mask for w, in struct.iter_unpack(fmt, raw))


# Original full source/shape contract remains unchanged; only the observation cut differs.
STAGES = (('layer9.input', [1,1,5120]), ('layer9.input_norm', [1,1,5120]),
          ('layer9.qkv', [1,1,10240]), ('layer9.z', [1,1,6144]),
          ('layer9.b', [1,1,48]), ('layer9.a', [1,1,48]),
          ('layer9.core', [48,128]), ('layer9.norm_gate', [48,128]), ('layer9.gated', [48,128]),
          ('layer9.gdn_out', [1,1,5120]), ('layer9.post_attention', [1,1,5120]),
          ('layer9.post_norm', [1,1,5120]), ('layer9.ffn_gate', [1,1,17408]),
          ('layer9.ffn_up', [1,1,17408]), ('layer9.ffn_gated', [1,1,17408]),
          ('layer9.ffn_down', [1,1,5120]))

def scope():
    return {'same_original_64_layer_object': True, 'completed_prefix_layers': 10,
            'used_source_roles': 135, 'initialized_logical_domains': 18,
            'full_hf_conv4': True, 'teacher': 11, 'repetitions': 2,
            'stopped_after_layer9_return_and_source_unload': True,
            'original_full_text_forward': False, 'final_norm_or_full_vocab_observed': False,
            'rust_numerical_parity_qualified': False, 'mkl_exp_producer_qualified': False,
            'state_root_native_serving_performance_qualified': False}

def state_specs(position, moment):
    require(position == 0 and moment in ('before','after'), 'initial-one prefix only')
    return [s for s in full_state_specs(position,moment) if s[0] <= 9]

def record_specs():
    for repeat in range(2):
        for layer,label,shape,dtype in [(None,'embedding',[1,1,5120],'BF16'),(None,'positions',[3,1,1],'I64')]:
            yield repeat,0,11,layer,label,shape,dtype
        for layer in range(9):
            yield repeat,0,11,layer,'layer_output',[1,1,5120],'BF16'
        for label,shape in STAGES:
            yield repeat,0,11,9,label,shape,'BF16'
        yield repeat,0,11,9,'layer_output',[1,1,5120],'BF16'
        for layer,label,shape,dtype in state_specs(0,'after'):
            yield repeat,0,11,layer,'after.'+label,shape,dtype

def prefix_bytes():
    return sum(extent(s[5],s[6]) for s in record_specs())
