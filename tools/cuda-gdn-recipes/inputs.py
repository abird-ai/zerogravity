"""Exact-byte imports and held passing original capture inputs."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
import sys
import types

from common import require, sha

HERE = Path(__file__).resolve().parent


def manifest(name):
    value = json.loads((HERE / name).read_text())
    require(type(value) is dict and bool(value), 'fixed nonempty source pin manifest')
    return value


def authenticate(held, root, name):
    for relative, pin in manifest(name).items():
        part = Path(relative)
        require(not part.is_absolute() and '..' not in part.parts, 'relative owned source pin')
        held.hold(root / part, expected_sha=pin['sha256'], expected_bytes=pin['bytes'])


def module(path, name, raw):
    value = types.ModuleType(name)
    value.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), value.__dict__)
    return value


def stamp(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@contextmanager
def reference_reader(root):
    """Bootstrap with stdlib; import only bytes authenticated on held FDs."""
    opened, sources = [], {}
    previous = {name: sys.modules.get(name) for name in ('contract', 'discovery')}
    try:
        pins = manifest('reference-pins.json')
        require(len(pins) == 14, 'complete frozen reference adapter cohort')
        for name, pin in pins.items():
            require(Path(name).name == name, 'flat exact reference source names')
            path = root / name
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            before = os.fstat(fd)
            opened.append((path, fd, stamp(before)))
            require(stat.S_ISREG(before.st_mode) and before.st_size == pin['bytes'] <= 1024**2,
                    'bounded original reference source size')
            raw = os.pread(fd, before.st_size + 1, 0)
            require(len(raw) == before.st_size and sha(raw) == pin['sha256']
                    and stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()),
                    'exact authenticated reference bytes before import')
            sources[name] = raw
        sys.modules['contract'] = module(root / 'contract.py', 'contract', sources['contract.py'])
        reference = module(root / 'discovery.py', 'discovery', sources['discovery.py'])
        sys.modules['discovery'] = reference
        yield reference
        for path, fd, before in opened:
            require(before == stamp(os.fstat(fd)) == stamp(path.lstat()), 'frozen reference source changed')
            require(sha(os.pread(fd, os.fstat(fd).st_size + 1, 0)) == pins[path.name]['sha256'],
                    'same exact reference import source through completion')
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        for _, fd, _ in opened:
            os.close(fd)


def renderer(held, path):
    pin = manifest('ancestor-pins.json')['renderer']
    held.hold(path, expected_sha=pin['sha256'], expected_bytes=pin['bytes'])
    return module(path, '_zg_original_gdn_renderer', held.content(path))


def prepared_capture(held, root, result, reference_root):
    prepared = held.document(root / 'prepared-package.json', result['prepared_package_sha256'], limit=1024**2)
    pins = manifest('reference-pins.json')
    stage = held.document(reference_root / 'capture-pins.json')
    expected = {str(Path(name).relative_to('package')): pin for name, pin in stage.items()
                if name.startswith('package/')}
    require(len(stage) == 50 and set(prepared) == set(expected) | {
        'reference_request.py', 'tokenize_prompt.py', 'request.json'}, 'complete full-ABI prepared source cohort')
    for relative, digest in prepared.items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                'relative prepared source path')
        held.hold(root / 'package' / relative, expected_sha=digest)
    for name, pin in expected.items():
        if name not in ('matrix_contract.py', 'validate_native.py'):
            held.hold(root / 'package' / name, expected_sha=pin['sha256'], expected_bytes=pin['bytes'])
    native = held.content(root / 'package/validate_native.py').decode()
    require(native.count('prompt_rows == PROMPT_ROWS') == 1
            and sha(native.replace('prompt_rows == PROMPT_ROWS', 'prompt_rows in (47,57,62,67,72)').encode())
            == expected['validate_native.py']['sha256'], 'only declared original native prompt guard changed')
    for staged, source in (('reference_request.py', 'contract.py'), ('matrix_contract.py', 'matrix_contract.py'),
                           ('tokenize_prompt.py', 'tokenize_prompt.py')):
        held.hold(root / 'package' / staged, expected_sha=pins[source]['sha256'])
    require(held.document(root / 'package/request.json', limit=1024**2) == result['tokenized_request'],
            'same actual tokenizer request in executed prepared package')


def accepted_capture(held, stack, path, digest, reference, reference_root):
    require(path.name == 'reference-result.json', 'explicit actual reference result')
    root = path.parent
    result = held.document(path, digest, limit=1024**2)
    require(result.get('schema') == 'zerogravity.qwen35-cuda-reference.result.v1'
            and result.get('mode') == result.get('ancestry_stage') == 'capture'
            and result.get('original_trajectory_validation') == 'PASS'
            and result.get('eight_calls_complete') is True and result.get('natural_calls') == 8
            and all(result.get(name) is False for name in (
                'native_recipes_qualified', 'model_execution_qualified', 'performance_qualified')),
            'passing complete actual packed reference; no source/model promotion')
    require(result['ancestor_manifest_sha256'] == sha(held.content(reference_root / 'capture-pins.json')),
            'same full-ABI source ancestry')
    request = reference.request(result['request'])
    require(request['prompt_tokens'] in (48, 49, 50), 'only fresh actual .8B48–50 additions')
    comparison = held.document(root / 'discovery-comparison.json', result['discovery_comparison_sha256'])
    require(comparison['status'] == 'PASS_COMPLETE_SAME_REQUEST_ORIGINAL_BYTES',
            'complete same-request raw/state/fullV comparison')
    reference.command_complete(held.document(root / 'comparison-command.json', limit=1024**2), held, root, 'comparison')
    discovery = stack.enter_context(reference.Discovery(Path(comparison['discovery']), request))
    require(result['tokenized_request'] == discovery.result['tokenized_request'], 'same original tokenizer operands')
    prepared_capture(held, root, result, reference_root)
    checked = reference.compare(discovery, root)
    for key in set(checked) | set(comparison):
        if key in ('held_discovery_inputs', 'held_capture_inputs'):
            def pins(rows):
                require(len(rows) == len({row['path'] for row in rows}), 'unique held comparison inputs')
                return {row['path']: (row['bytes'], row['sha256']) for row in rows}
            require(pins(checked[key]) == pins(comparison[key]), 'same complete held input identities')
        else:
            require(checked[key] == comparison[key], 'same actual complete source-byte comparison')
    for pin in comparison['held_capture_inputs']:
        item = Path(pin['path'])
        require(item.is_absolute() and item.is_relative_to(root) and '..' not in item.parts,
                'this capture owns the held capture inputs')
        held.hold(item, expected_sha=pin['sha256'], expected_bytes=pin['bytes'])
    trajectory = held.document(root / 'trajectory-validation.json', result['trajectory_sha256'])
    require(trajectory['model'] == '0.8b' and trajectory['prompt_rows'] == request['prompt_tokens']
            and trajectory['eight_calls_complete'] is True and trajectory['outputs'] == 8
            and trajectory['original']['source_receipt_sha256'] == reference.SOURCE_MANIFEST,
            'actual complete .8B source/request trajectory')
    return root, request['prompt_tokens'], trajectory, discovery
