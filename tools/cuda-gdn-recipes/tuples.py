"""Actual source/launch tuples and the immutable original subset."""
import ast
import re
from pathlib import Path

from common import canonical, require, sha
from inputs import authenticate, manifest

REL = Path('crates/zerogravity-gpu/src/cuda/gdn/text/observed.rs')
MODEL = Path('crates/zerogravity-model-qwen35/src/source/binding/window_model/prepare/gdn.rs')
STAGES = {'_causal_conv1d_fwd_kernel': 'Conv', '_fused_post_conv_kernel': 'Prep',
          'chunk_local_cumsum_scalar_kernel': 'Cumsum', 'chunk_scaled_dot_kkt_fwd_kernel': 'Kkt',
          'merge_16x16_to_64x64_inverse_kernel': 'Solve', 'recompute_w_u_fwd_kernel': 'Recompute',
          'chunk_gated_delta_rule_fwd_kernel_h_blockdim64': 'H', 'chunk_fwd_kernel_o': 'Output',
          'layer_norm_fwd_kernel': 'Gated'}
CACHED = {'_causal_conv1d_update_kernel': 'Conv',
          'fused_recurrent_gated_delta_rule_packed_decode_kernel': 'Decode', 'layer_norm_fwd_kernel': 'Gated'}


def recipe(profile, tokens, decode, pins):
    return dict(profile=profile, input_tokens=tokens, is_decode=decode, pins=pins,
                recipe_sha256=sha(canonical(dict(tokens=tokens, decode=decode, stages=pins))))


def catalogue(records, profile):
    return sha(canonical([{key: row[key] for key in (
        'profile', 'input_tokens', 'is_decode', 'recipe_sha256', 'pins')}
        for row in records if row['profile'] == profile]))


def pin(stage, descriptor, file, raw):
    require(sha(raw) == descriptor['source_sha256'], 'actual rendered CUDA source digest')
    return dict(stage=stage, entry=descriptor['entry'], file=file, threads=descriptor['threads'],
                grid=descriptor['grid'], shared=descriptor['shared'], ptx=descriptor['original_ptx_sha256'],
                source_sha256=descriptor['source_sha256'])


def scalar(body, name):
    match = re.search(r'\b' + name + r':\s*([^,\n]+)', body)
    require(match is not None, 'actual literal source field: ' + name)
    return ast.literal_eval(match[1].strip())


def baseline(held, root, original_path):
    authenticate(held, root, 'baseline-pins.json')
    expected = manifest('ancestor-pins.json')['original_tuples']
    original = held.document(original_path, expected['sha256'], limit=expected['bytes'])
    require(len(original) == 12, 'complete original12capture/96tuple source archive')
    text = held.content(root / REL).decode()
    heads = list(re.finditer(r'\((\d+),\s*GdnPhase::(?:TextPrefill\((\d+)\)|CachedDecode)\)\s*=>', text))
    require(len(heads) == 12, 'actual unchanged native selector membership')
    actual = []
    for cell, head in enumerate(heads):
        require(int(head[1]) == cell, 'original native cell order')
        stop = heads[cell + 1].start() if cell + 1 < len(heads) else text.index('pub(super) const CATALOGUE:')
        pins = []
        for item in re.finditer(r'(\w+)\s*=>\s*assets::Asset\s*\{([^}]+)\}', text[head.end():stop]):
            body = item[2]
            file = re.search(r'source:\s*include_str!\("([^"]+)"\)', body)[1]
            require(not Path(file).is_absolute() and '..' not in Path(file).parts, 'relative original CU source')
            raw = held.content(root / REL.parent / file)
            pins.append(dict(stage=item[1], entry=scalar(body, 'entry'), file=file,
                threads=scalar(body, 'threads'), grid=ast.literal_eval(re.search(r'grid:\s*(\[[^]]+\])', body)[1]),
                shared=scalar(body, 'shared'), ptx=scalar(body, 'ptx_sha256'), source_sha256=sha(raw)))
        decode = head[2] is None
        require(len(pins) == (3 if decode else 9), 'complete original GDN stages')
        row = recipe('Qwen08' if cell < 6 else 'Qwen2B', 1 if decode else int(head[2]), decode, pins)
        require(all(row[key] == original[cell][key] for key in row), 'every old tuple/recipe byte preserved')
        actual.append(row)
    require(sum(len(row['pins']) for row in actual) == 96, 'all original96 tuples resolved')
    gpu = re.findall(r'Capture\s*\{\s*recipe_sha256:\s*"([a-f0-9]{64})",\s*catalogue_sha256:\s*"([a-f0-9]{64})",\s*input_tokens:\s*(\d+),\s*is_decode:\s*(true|false)', text)
    model = re.findall(r'GdnKernelCapture\s*\{\s*recipe_sha256:\s*"([a-f0-9]{64})",\s*input_tokens:\s*(\d+),\s*is_decode:\s*(true|false)', held.content(root / MODEL).decode())
    require(len(gpu) == len(model) == 12, 'original GPU/model recipe membership')
    for row, g, m, old in zip(actual, gpu, model, original):
        family = catalogue(actual, row['profile'])
        require(g == (row['recipe_sha256'], family, str(row['input_tokens']), str(row['is_decode']).lower())
                and m == (g[0], g[2], g[3]) and family == old['catalogue_sha256'],
                'recomputed actual source catalogue agrees with GPU/model/archive pins')
    return original


def rendered_signature(descriptor):
    return {name: descriptor[name] for name in (
        'entry', 'threads', 'shared', 'grid', 'scalars', 'original_ptx_sha256', 'source_sha256')}, [
        {'name': binding['name'], 'type': binding['type']} for binding in descriptor['bindings']]


